/*
** Copyright (c) 2026 LunarG, Inc.
**
** Permission is hereby granted, free of charge, to any person obtaining a
** copy of this software and associated documentation files (the "Software"),
** to deal in the Software without restriction, including without limitation
** the rights to use, copy, modify, merge, publish, distribute, sublicense,
** and/or sell copies of the Software, and to permit persons to whom the
** Software is furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
** FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
** DEALINGS IN THE SOFTWARE.
*/

// Tests VulkanRebindAllocator through its public API, with the real VMA and a fake Vulkan device underneath.

#include <catch2/catch.hpp>

#include "decode/vulkan_object_info.h"
#include "decode/vulkan_rebind_allocator.h"
#include "format/format_util.h"
#include "util/logging.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

using gfxrecon::decode::VulkanRebindAllocator;
using ResourceData = gfxrecon::decode::VulkanResourceAllocator::ResourceData;
using MemoryData   = gfxrecon::decode::VulkanResourceAllocator::MemoryData;

namespace
{

constexpr VkMemoryPropertyFlags kDeviceLocal = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
constexpr VkMemoryPropertyFlags kHostCoherent =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr VkMemoryPropertyFlags kHostCached     = kHostCoherent | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
constexpr VkMemoryPropertyFlags kLazy           = kDeviceLocal | VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
constexpr VkMemoryPropertyFlags kDeviceLocalAmd = kDeviceLocal | VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD;
constexpr VkMemoryPropertyFlags kProtected      = kDeviceLocal | VK_MEMORY_PROPERTY_PROTECTED_BIT;

// Replay memory types, ordered so that VMA picks a different type for each memory usage.
// The plain type comes first, so GPU_ONLY lands on it when no device-local type is allowed.
constexpr uint32_t     kReplayPlain         = 0;
constexpr uint32_t     kReplayCpuOnly       = 1;
constexpr uint32_t     kReplayGpuOnly       = 2;
constexpr uint32_t     kReplayCpuToGpu      = 3;
constexpr uint32_t     kReplayGpuToCpu      = 4;
constexpr uint32_t     kReplayLazy          = 5;
constexpr uint32_t     kReplayProtected     = 6;
constexpr uint32_t     kAllUnprotectedTypes = 0x3f;
constexpr VkDeviceSize kHeapSize            = VkDeviceSize{ 256 } << 20;

// Capture memory types. Their indices differ from the replay indices of the same flags.
constexpr uint32_t kCaptureHostCached      = 0;
constexpr uint32_t kCaptureDeviceLocal     = 1;
constexpr uint32_t kCaptureLazy            = 2;
constexpr uint32_t kCaptureHostCoherent    = 3;
constexpr uint32_t kCaptureDeviceLocalAmd  = 4;
constexpr uint32_t kCaptureDeviceLocalHost = 5;

constexpr VkDeviceSize kImageSize       = 65536;
constexpr VkDeviceSize kImageAlignment  = 4096;
constexpr VkDeviceSize kBufferAlignment = 256;

VkPhysicalDeviceMemoryProperties MakeMemoryProperties(const std::vector<VkMemoryPropertyFlags>& type_flags)
{
    VkPhysicalDeviceMemoryProperties properties{};
    properties.memoryHeapCount = 2;
    properties.memoryHeaps[0]  = { kHeapSize, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT };
    properties.memoryHeaps[1]  = { kHeapSize, 0 };

    properties.memoryTypeCount = static_cast<uint32_t>(type_flags.size());
    for (uint32_t i = 0; i < properties.memoryTypeCount; ++i)
    {
        const bool device_local   = (type_flags[i] & kDeviceLocal) != 0;
        properties.memoryTypes[i] = { type_flags[i], device_local ? 0u : 1u };
    }
    return properties;
}

const VkPhysicalDeviceMemoryProperties kReplayMemoryProperties = MakeMemoryProperties(
    { 0, kHostCoherent, kDeviceLocal, kDeviceLocal | kHostCoherent, kHostCached, kLazy, kProtected });

const VkPhysicalDeviceMemoryProperties kCaptureMemoryProperties = MakeMemoryProperties(
    { kHostCached, kDeviceLocal, kLazy, kHostCoherent, kDeviceLocalAmd, kDeviceLocal | kHostCoherent });

struct AllocateRecord
{
    VkDeviceMemory memory{ VK_NULL_HANDLE };
    VkDeviceSize   size{ 0 };
    uint32_t       memory_type{ 0 };
    VkImage        dedicated_image{ VK_NULL_HANDLE };
    VkBuffer       dedicated_buffer{ VK_NULL_HANDLE };
};

struct BindRecord
{
    uint64_t       resource{ 0 };
    VkDeviceMemory memory{ VK_NULL_HANDLE };
    VkDeviceSize   offset{ 0 };
};

// Fake Vulkan device. Records the calls it receives and answers queries from the settings below.
struct FakeDevice
{
    uint32_t memory_type_bits{ kAllUnprotectedTypes };
    bool     image_requires_dedicated{ false };

    std::vector<VkDataGraphPipelineSessionBindPointRequirementARM> session_bind_points;
    VkDeviceSize                                                   session_memory_size{ 4096 };
    VkResult                                                       session_bind_result{ VK_SUCCESS };

    std::vector<AllocateRecord> allocations;
    std::vector<VkDeviceMemory> frees;
    std::vector<BindRecord>     binds;
    std::vector<VkCommandPool>  created_pools;
    std::vector<VkCommandPool>  destroyed_pools;

    std::vector<VkDataGraphPipelineSessionARM>                       created_sessions;
    std::vector<VkDataGraphPipelineSessionARM>                       destroyed_sessions;
    std::vector<VkDataGraphPipelineSessionMemoryRequirementsInfoARM> session_memory_queries;
    std::vector<VkBindDataGraphPipelineSessionMemoryInfoARM>         session_binds;
    uint32_t                                                         session_bind_calls{ 0 };

    std::unordered_map<VkBuffer, VkDeviceSize> buffer_sizes;
    uint64_t                                   next_handle{ 0x1000 };

    template <typename T>
    T NewHandle()
    {
        return gfxrecon::format::FromHandleId<T>(next_handle++);
    }

    BindRecord BindOf(uint64_t resource) const
    {
        auto it = std::find_if(
            binds.rbegin(), binds.rend(), [resource](const BindRecord& bind) { return bind.resource == resource; });
        REQUIRE(it != binds.rend());
        return *it;
    }

    AllocateRecord AllocationOf(VkDeviceMemory memory) const
    {
        auto it = std::find_if(allocations.begin(), allocations.end(), [memory](const AllocateRecord& allocation) {
            return allocation.memory == memory;
        });
        REQUIRE(it != allocations.end());
        return *it;
    }
};

FakeDevice* g_fake = nullptr;

const VkBaseOutStructure* FindInChain(const void* next, VkStructureType type)
{
    for (auto entry = static_cast<const VkBaseOutStructure*>(next); entry != nullptr; entry = entry->pNext)
    {
        if (entry->sType == type)
        {
            return entry;
        }
    }
    return nullptr;
}

VKAPI_ATTR void VKAPI_CALL FakeGetPhysicalDeviceProperties(VkPhysicalDevice, VkPhysicalDeviceProperties* properties)
{
    *properties                                 = {};
    properties->apiVersion                      = VK_API_VERSION_1_0;
    properties->deviceType                      = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    properties->limits.bufferImageGranularity   = 1;
    properties->limits.nonCoherentAtomSize      = 64;
    properties->limits.maxMemoryAllocationCount = 4096;
}

VKAPI_ATTR void VKAPI_CALL FakeGetPhysicalDeviceMemoryProperties(VkPhysicalDevice,
                                                                 VkPhysicalDeviceMemoryProperties* properties)
{
    *properties = kReplayMemoryProperties;
}

VKAPI_ATTR void VKAPI_CALL FakeGetPhysicalDeviceQueueFamilyProperties(VkPhysicalDevice,
                                                                      uint32_t*                count,
                                                                      VkQueueFamilyProperties* properties)
{
    if (properties != nullptr)
    {
        properties[0]            = {};
        properties[0].queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT;
        properties[0].queueCount = 1;
    }
    *count = 1;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeAllocateMemory(VkDevice,
                                                  const VkMemoryAllocateInfo* info,
                                                  const VkAllocationCallbacks*,
                                                  VkDeviceMemory* memory)
{
    *memory = g_fake->NewHandle<VkDeviceMemory>();

    AllocateRecord record{ *memory, info->allocationSize, info->memoryTypeIndex };
    if (auto dedicated = reinterpret_cast<const VkMemoryDedicatedAllocateInfo*>(
            FindInChain(info->pNext, VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO)))
    {
        record.dedicated_image  = dedicated->image;
        record.dedicated_buffer = dedicated->buffer;
    }
    g_fake->allocations.push_back(record);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeFreeMemory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*)
{
    g_fake->frees.push_back(memory);
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateBuffer(VkDevice,
                                                const VkBufferCreateInfo* info,
                                                const VkAllocationCallbacks*,
                                                VkBuffer* buffer)
{
    *buffer                       = g_fake->NewHandle<VkBuffer>();
    g_fake->buffer_sizes[*buffer] = info->size;
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyBuffer(VkDevice, VkBuffer, const VkAllocationCallbacks*) {}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateImage(VkDevice,
                                               const VkImageCreateInfo*,
                                               const VkAllocationCallbacks*,
                                               VkImage* image)
{
    *image = g_fake->NewHandle<VkImage>();
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyImage(VkDevice, VkImage, const VkAllocationCallbacks*) {}

VKAPI_ATTR void VKAPI_CALL FakeGetBufferMemoryRequirements(VkDevice,
                                                           VkBuffer              buffer,
                                                           VkMemoryRequirements* requirements)
{
    const VkDeviceSize size      = g_fake->buffer_sizes.at(buffer);
    requirements->size           = (size + kBufferAlignment - 1) / kBufferAlignment * kBufferAlignment;
    requirements->alignment      = kBufferAlignment;
    requirements->memoryTypeBits = g_fake->memory_type_bits;
}

VKAPI_ATTR void VKAPI_CALL FakeGetImageMemoryRequirements(VkDevice, VkImage, VkMemoryRequirements* requirements)
{
    requirements->size           = kImageSize;
    requirements->alignment      = kImageAlignment;
    requirements->memoryTypeBits = g_fake->memory_type_bits;
}

VKAPI_ATTR void VKAPI_CALL FakeGetBufferMemoryRequirements2(VkDevice,
                                                            const VkBufferMemoryRequirementsInfo2* info,
                                                            VkMemoryRequirements2*                 requirements)
{
    FakeGetBufferMemoryRequirements(VK_NULL_HANDLE, info->buffer, &requirements->memoryRequirements);
}

VKAPI_ATTR void VKAPI_CALL FakeGetImageMemoryRequirements2(VkDevice,
                                                           const VkImageMemoryRequirementsInfo2* info,
                                                           VkMemoryRequirements2*                requirements)
{
    FakeGetImageMemoryRequirements(VK_NULL_HANDLE, info->image, &requirements->memoryRequirements);

    auto dedicated = reinterpret_cast<VkMemoryDedicatedRequirements*>(const_cast<VkBaseOutStructure*>(
        FindInChain(requirements->pNext, VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS)));
    if (dedicated != nullptr)
    {
        dedicated->requiresDedicatedAllocation = g_fake->image_requires_dedicated ? VK_TRUE : VK_FALSE;
        dedicated->prefersDedicatedAllocation  = dedicated->requiresDedicatedAllocation;
    }
}

VKAPI_ATTR VkResult VKAPI_CALL FakeBindBufferMemory(VkDevice,
                                                    VkBuffer       buffer,
                                                    VkDeviceMemory memory,
                                                    VkDeviceSize   offset)
{
    g_fake->binds.push_back({ VK_HANDLE_TO_UINT64(buffer), memory, offset });
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeBindImageMemory(VkDevice, VkImage image, VkDeviceMemory memory, VkDeviceSize offset)
{
    g_fake->binds.push_back({ VK_HANDLE_TO_UINT64(image), memory, offset });
    return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateCommandPool(VkDevice,
                                                     const VkCommandPoolCreateInfo*,
                                                     const VkAllocationCallbacks*,
                                                     VkCommandPool* pool)
{
    *pool = g_fake->NewHandle<VkCommandPool>();
    g_fake->created_pools.push_back(*pool);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyCommandPool(VkDevice, VkCommandPool pool, const VkAllocationCallbacks*)
{
    g_fake->destroyed_pools.push_back(pool);
}

VKAPI_ATTR void VKAPI_CALL FakeGetDeviceQueue(VkDevice, uint32_t, uint32_t, VkQueue* queue)
{
    *queue = g_fake->NewHandle<VkQueue>();
}

VKAPI_ATTR VkResult VKAPI_CALL FakeCreateDataGraphPipelineSession(VkDevice,
                                                                  const VkDataGraphPipelineSessionCreateInfoARM*,
                                                                  const VkAllocationCallbacks*,
                                                                  VkDataGraphPipelineSessionARM* session)
{
    *session = g_fake->NewHandle<VkDataGraphPipelineSessionARM>();
    g_fake->created_sessions.push_back(*session);
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeDestroyDataGraphPipelineSession(VkDevice,
                                                               VkDataGraphPipelineSessionARM session,
                                                               const VkAllocationCallbacks*)
{
    g_fake->destroyed_sessions.push_back(session);
}

VKAPI_ATTR VkResult VKAPI_CALL
FakeGetDataGraphPipelineSessionBindPointRequirements(VkDevice,
                                                     const VkDataGraphPipelineSessionBindPointRequirementsInfoARM*,
                                                     uint32_t*                                          count,
                                                     VkDataGraphPipelineSessionBindPointRequirementARM* requirements)
{
    const auto& bind_points = g_fake->session_bind_points;
    if (requirements != nullptr)
    {
        REQUIRE(*count == bind_points.size());
        for (size_t i = 0; i < bind_points.size(); ++i)
        {
            requirements[i].bindPoint     = bind_points[i].bindPoint;
            requirements[i].bindPointType = bind_points[i].bindPointType;
            requirements[i].numObjects    = bind_points[i].numObjects;
        }
    }
    *count = static_cast<uint32_t>(bind_points.size());
    return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL FakeGetDataGraphPipelineSessionMemoryRequirements(
    VkDevice, const VkDataGraphPipelineSessionMemoryRequirementsInfoARM* info, VkMemoryRequirements2* requirements)
{
    g_fake->session_memory_queries.push_back(*info);
    requirements->memoryRequirements.size           = g_fake->session_memory_size;
    requirements->memoryRequirements.alignment      = kBufferAlignment;
    requirements->memoryRequirements.memoryTypeBits = g_fake->memory_type_bits;
}

VKAPI_ATTR VkResult VKAPI_CALL FakeBindDataGraphPipelineSessionMemory(
    VkDevice, uint32_t count, const VkBindDataGraphPipelineSessionMemoryInfoARM* infos)
{
    ++g_fake->session_bind_calls;
    g_fake->session_binds.insert(g_fake->session_binds.end(), infos, infos + count);
    return g_fake->session_bind_result;
}

// A rebind allocator initialized on the fake device. Destroys what the test left behind.
class RebindFixture
{
  public:
    explicit RebindFixture(const std::vector<std::string>& enabled_device_extensions = {},
                           VkPhysicalDeviceType            capture_device_type = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
    {
        gfxrecon::util::Log::Init(gfxrecon::util::LoggingSeverity::kError);
        g_fake = &fake_;

        instance_table_.GetPhysicalDeviceProperties            = FakeGetPhysicalDeviceProperties;
        instance_table_.GetPhysicalDeviceMemoryProperties      = FakeGetPhysicalDeviceMemoryProperties;
        instance_table_.GetPhysicalDeviceQueueFamilyProperties = FakeGetPhysicalDeviceQueueFamilyProperties;

        device_table_.AllocateMemory                  = FakeAllocateMemory;
        device_table_.FreeMemory                      = FakeFreeMemory;
        device_table_.CreateBuffer                    = FakeCreateBuffer;
        device_table_.DestroyBuffer                   = FakeDestroyBuffer;
        device_table_.CreateImage                     = FakeCreateImage;
        device_table_.DestroyImage                    = FakeDestroyImage;
        device_table_.GetBufferMemoryRequirements     = FakeGetBufferMemoryRequirements;
        device_table_.GetImageMemoryRequirements      = FakeGetImageMemoryRequirements;
        device_table_.GetBufferMemoryRequirements2KHR = FakeGetBufferMemoryRequirements2;
        device_table_.GetImageMemoryRequirements2KHR  = FakeGetImageMemoryRequirements2;
        device_table_.BindBufferMemory                = FakeBindBufferMemory;
        device_table_.BindImageMemory                 = FakeBindImageMemory;
        device_table_.CreateCommandPool               = FakeCreateCommandPool;
        device_table_.DestroyCommandPool              = FakeDestroyCommandPool;
        device_table_.GetDeviceQueue                  = FakeGetDeviceQueue;

        device_table_.CreateDataGraphPipelineSessionARM  = FakeCreateDataGraphPipelineSession;
        device_table_.DestroyDataGraphPipelineSessionARM = FakeDestroyDataGraphPipelineSession;
        device_table_.GetDataGraphPipelineSessionBindPointRequirementsARM =
            FakeGetDataGraphPipelineSessionBindPointRequirements;
        device_table_.GetDataGraphPipelineSessionMemoryRequirementsARM =
            FakeGetDataGraphPipelineSessionMemoryRequirements;
        device_table_.BindDataGraphPipelineSessionMemoryARM = FakeBindDataGraphPipelineSessionMemory;

        VkPhysicalDeviceProperties replay_properties{};
        FakeGetPhysicalDeviceProperties(VK_NULL_HANDLE, &replay_properties);
        replay_device_info_.properties        = replay_properties;
        replay_device_info_.memory_properties = kReplayMemoryProperties;

        physical_device_info_.handle                    = fake_.NewHandle<VkPhysicalDevice>();
        physical_device_info_.parent                    = fake_.NewHandle<VkInstance>();
        physical_device_info_.capture_device_type       = capture_device_type;
        physical_device_info_.capture_memory_properties = kCaptureMemoryProperties;
        physical_device_info_.replay_device_info        = &replay_device_info_;

        const float             queue_priority = 1.0f;
        VkDeviceQueueCreateInfo queue_info{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        queue_info.queueFamilyIndex = 0;
        queue_info.queueCount       = 1;
        queue_info.pQueuePriorities = &queue_priority;

        VkDeviceCreateInfo device_info{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos    = &queue_info;

        REQUIRE(allocator_.Initialize(&physical_device_info_,
                                      fake_.NewHandle<VkDevice>(),
                                      device_info,
                                      enabled_device_extensions,
                                      instance_table_,
                                      &device_table_) == VK_SUCCESS);
    }

    ~RebindFixture()
    {
        Teardown();
        g_fake = nullptr;
        gfxrecon::util::Log::Release();
    }

    RebindFixture(const RebindFixture&)            = delete;
    RebindFixture& operator=(const RebindFixture&) = delete;

    FakeDevice&            fake() { return fake_; }
    VulkanRebindAllocator& allocator() { return allocator_; }

    // Creates an image and reports the captured memory requirements for it.
    VkImage CreateImage(VkImageTiling tiling, VkImageUsageFlags usage, ResourceData* data)
    {
        VkImageCreateInfo info{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
        info.imageType   = VK_IMAGE_TYPE_2D;
        info.format      = VK_FORMAT_R8G8B8A8_UNORM;
        info.extent      = { 64, 64, 1 };
        info.mipLevels   = 1;
        info.arrayLayers = 1;
        info.samples     = VK_SAMPLE_COUNT_1_BIT;
        info.tiling      = tiling;
        info.usage       = usage;

        VkImage image = VK_NULL_HANDLE;
        REQUIRE(allocator_.CreateImage(&info, nullptr, next_capture_id_++, &image, data) == VK_SUCCESS);

        VkMemoryRequirements capture_requirements{ kImageSize, kImageAlignment, kAllUnprotectedTypes };
        allocator_.GetImageMemoryRequirements(image, &capture_requirements, *data);

        images_.push_back({ image, *data });
        return image;
    }

    // Creates a buffer and reports the captured memory requirements for it.
    VkBuffer CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, ResourceData* data)
    {
        VkBufferCreateInfo info{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        info.size  = size;
        info.usage = usage;

        VkBuffer buffer = VK_NULL_HANDLE;
        REQUIRE(allocator_.CreateBuffer(&info, nullptr, next_capture_id_++, &buffer, data) == VK_SUCCESS);

        VkMemoryRequirements capture_requirements{ size, kBufferAlignment, kAllUnprotectedTypes };
        allocator_.GetBufferMemoryRequirements(buffer, &capture_requirements, *data);

        buffers_.push_back({ buffer, *data });
        return buffer;
    }

    // Allocates captured memory. Rebind defers the real allocation to the first bind.
    VkDeviceMemory AllocateMemory(VkDeviceSize size, uint32_t capture_memory_type, MemoryData* data)
    {
        VkMemoryAllocateInfo info{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        info.allocationSize  = size;
        info.memoryTypeIndex = capture_memory_type;

        VkDeviceMemory memory = VK_NULL_HANDLE;
        REQUIRE(allocator_.AllocateMemory(&info, nullptr, next_capture_id_++, &memory, data) == VK_SUCCESS);

        memories_.push_back({ memory, *data });
        return memory;
    }

    // Creates a data graph session. Rebind allocates and binds the session memory here.
    VkResult CreateSession(VkDataGraphPipelineSessionCreateFlagsARM flags,
                           VkDataGraphPipelineSessionARM*           session,
                           ResourceData*                            data)
    {
        VkDataGraphPipelineSessionCreateInfoARM info{ VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_CREATE_INFO_ARM };
        info.flags = flags;

        const VkResult result =
            allocator_.CreateDataGraphPipelineSession(&info, nullptr, next_capture_id_++, session, data);
        if (result == VK_SUCCESS)
        {
            sessions_.push_back({ *session, *data });
        }
        return result;
    }

    void DestroySession(VkDataGraphPipelineSessionARM session, ResourceData data)
    {
        allocator_.DestroyDataGraphPipelineSession(session, nullptr, data);
        sessions_.erase(std::find(sessions_.begin(), sessions_.end(), std::make_pair(session, data)));
    }

    // Destroys all resources, frees all memory, and destroys the allocator.
    void Teardown()
    {
        if (destroyed_)
        {
            return;
        }
        destroyed_ = true;

        for (const auto& [session, data] : sessions_)
        {
            allocator_.DestroyDataGraphPipelineSession(session, nullptr, data);
        }
        for (const auto& [image, data] : images_)
        {
            allocator_.DestroyImage(image, nullptr, data);
        }
        for (const auto& [buffer, data] : buffers_)
        {
            allocator_.DestroyBuffer(buffer, nullptr, data);
        }
        for (const auto& [memory, data] : memories_)
        {
            allocator_.FreeMemory(memory, nullptr, data);
        }
        allocator_.Destroy();
    }

  private:
    FakeDevice                                 fake_;
    gfxrecon::graphics::VulkanInstanceTable    instance_table_;
    gfxrecon::graphics::VulkanDeviceTable      device_table_;
    gfxrecon::decode::VulkanReplayDeviceInfo   replay_device_info_;
    gfxrecon::decode::VulkanPhysicalDeviceInfo physical_device_info_;
    VulkanRebindAllocator                      allocator_;
    gfxrecon::format::HandleId                 next_capture_id_{ 1 };
    bool                                       destroyed_{ false };

    std::vector<std::pair<VkImage, ResourceData>>      images_;
    std::vector<std::pair<VkBuffer, ResourceData>>     buffers_;
    std::vector<std::pair<VkDeviceMemory, MemoryData>> memories_;

    std::vector<std::pair<VkDataGraphPipelineSessionARM, ResourceData>> sessions_;
};

} // namespace

TEST_CASE("Rebind translates the captured memory type to a replay memory type", "[rebind]")
{
    RebindFixture fixture;

    ResourceData buffer_data = 0;
    MemoryData   memory_data = 0;
    VkBuffer     buffer =
        fixture.CreateBuffer(4096, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &buffer_data);
    VkDeviceMemory memory = fixture.AllocateMemory(4096, kCaptureDeviceLocal, &memory_data);

    VkMemoryPropertyFlags bind_properties = 0;
    REQUIRE(fixture.allocator().BindBufferMemory(buffer, memory, 0, buffer_data, memory_data, &bind_properties) ==
            VK_SUCCESS);

    const BindRecord bind = fixture.fake().BindOf(VK_HANDLE_TO_UINT64(buffer));
    CHECK(fixture.fake().AllocationOf(bind.memory).memory_type == kReplayGpuOnly);
    CHECK(bind_properties == kDeviceLocal);
}

TEST_CASE("Rebind binds at the replay offset, not the captured offset", "[rebind]")
{
    RebindFixture fixture;

    ResourceData   image_data  = 0;
    MemoryData     memory_data = 0;
    VkImage        image       = fixture.CreateImage(VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT, &image_data);
    VkDeviceMemory memory      = fixture.AllocateMemory(2 * kImageSize, kCaptureDeviceLocal, &memory_data);

    VkMemoryPropertyFlags bind_properties = 0;
    REQUIRE(fixture.allocator().BindImageMemory(
                image, memory, kImageAlignment, image_data, memory_data, &bind_properties) == VK_SUCCESS);

    // The first allocation from a fresh allocator starts its VMA block.
    CHECK(fixture.fake().BindOf(VK_HANDLE_TO_UINT64(image)).offset == 0);
}

TEST_CASE("Rebind picks the image memory usage from tiling, usage and captured memory type", "[rebind]")
{
    struct Row
    {
        const char*       name;
        VkImageTiling     tiling;
        VkImageUsageFlags usage;
        uint32_t          capture_memory_type;
        uint32_t          replay_type_bits;
        uint32_t          expected_replay_type;
    };

    const VkImageUsageFlags kLinearMixedUsage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                                VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;

    const Row rows[] = {
        { "optimal, device local",
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_SAMPLED_BIT,
          kCaptureDeviceLocal,
          kAllUnprotectedTypes,
          kReplayGpuOnly },
        { "optimal, host visible",
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_SAMPLED_BIT,
          kCaptureHostCoherent,
          kAllUnprotectedTypes,
          kReplayCpuToGpu },
        { "optimal, host cached",
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_SAMPLED_BIT,
          kCaptureHostCached,
          kAllUnprotectedTypes,
          kReplayGpuToCpu },
        { "linear, transfer src only",
          VK_IMAGE_TILING_LINEAR,
          VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
          kCaptureHostCoherent,
          kAllUnprotectedTypes,
          kReplayCpuOnly },
        { "linear, transfer dst only",
          VK_IMAGE_TILING_LINEAR,
          VK_IMAGE_USAGE_TRANSFER_DST_BIT,
          kCaptureHostCoherent,
          kAllUnprotectedTypes,
          kReplayGpuToCpu },
        { "linear, device local",
          VK_IMAGE_TILING_LINEAR,
          VK_IMAGE_USAGE_SAMPLED_BIT,
          kCaptureDeviceLocal,
          kAllUnprotectedTypes,
          kReplayGpuOnly },
        { "transient, lazily allocated",
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
          kCaptureLazy,
          kAllUnprotectedTypes,
          kReplayLazy },
        { "host cached, but replay allows no host visible type",
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_SAMPLED_BIT,
          kCaptureHostCached,
          (1u << kReplayGpuOnly) | (1u << kReplayLazy),
          kReplayGpuOnly },
        { "linear, mixed usage",
          VK_IMAGE_TILING_LINEAR,
          VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
          kCaptureHostCoherent,
          kAllUnprotectedTypes,
          kReplayCpuToGpu },
        { "optimal, device local and host visible",
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_SAMPLED_BIT,
          kCaptureDeviceLocalHost,
          kAllUnprotectedTypes,
          kReplayGpuOnly },
        { "linear, mixed usage, device local and host visible",
          VK_IMAGE_TILING_LINEAR,
          kLinearMixedUsage,
          kCaptureDeviceLocalHost,
          kAllUnprotectedTypes,
          kReplayCpuToGpu },
        { "device local, but replay allows no device local type",
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_SAMPLED_BIT,
          kCaptureDeviceLocal,
          (1u << kReplayPlain) | (1u << kReplayCpuOnly),
          kReplayCpuOnly },
        { "host visible, but replay allows no host visible type",
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_SAMPLED_BIT,
          kCaptureHostCoherent,
          1u << kReplayGpuOnly,
          kReplayGpuOnly },
        { "transient, but replay allows no lazily allocated type",
          VK_IMAGE_TILING_OPTIMAL,
          VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
          kCaptureLazy,
          1u << kReplayGpuOnly,
          kReplayGpuOnly },
    };

    for (const Row& row : rows)
    {
        CAPTURE(row.name);

        RebindFixture fixture;
        fixture.fake().memory_type_bits = row.replay_type_bits;

        ResourceData   image_data  = 0;
        MemoryData     memory_data = 0;
        VkImage        image       = fixture.CreateImage(row.tiling, row.usage, &image_data);
        VkDeviceMemory memory      = fixture.AllocateMemory(kImageSize, row.capture_memory_type, &memory_data);

        // CHECK rather than REQUIRE, so a failing row does not hide the rows after it.
        VkMemoryPropertyFlags bind_properties = 0;
        const VkResult        result =
            fixture.allocator().BindImageMemory(image, memory, 0, image_data, memory_data, &bind_properties);
        CHECK(result == VK_SUCCESS);
        if (result != VK_SUCCESS)
        {
            continue;
        }

        const BindRecord bind = fixture.fake().BindOf(VK_HANDLE_TO_UINT64(image));
        CHECK(fixture.fake().AllocationOf(bind.memory).memory_type == row.expected_replay_type);
    }
}

TEST_CASE("Rebind picks the buffer memory usage from usage, captured memory type and capture device", "[rebind]")
{
    struct Row
    {
        const char*          name;
        VkPhysicalDeviceType capture_device_type;
        VkBufferUsageFlags   usage;
        uint32_t             capture_memory_type;
        uint32_t             expected_replay_type;
    };

    constexpr VkPhysicalDeviceType kIntegrated = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    constexpr VkPhysicalDeviceType kDiscrete   = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;

    const Row rows[] = {
        { "integrated, transfer dst, device local and host visible",
          kIntegrated,
          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          kCaptureDeviceLocalHost,
          kReplayGpuOnly },
        { "discrete, transfer dst, device local and host visible",
          kDiscrete,
          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          kCaptureDeviceLocalHost,
          kReplayCpuToGpu },
        { "integrated, uniform and transfer dst, device local and host visible",
          kIntegrated,
          VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          kCaptureDeviceLocalHost,
          kReplayCpuToGpu },
        { "transfer src only", kDiscrete, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kCaptureHostCoherent, kReplayCpuOnly },
        { "transfer dst only", kDiscrete, VK_BUFFER_USAGE_TRANSFER_DST_BIT, kCaptureHostCoherent, kReplayGpuToCpu },
        { "host cached", kDiscrete, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, kCaptureHostCached, kReplayGpuToCpu },
    };

    for (const Row& row : rows)
    {
        CAPTURE(row.name);

        RebindFixture fixture({}, row.capture_device_type);

        ResourceData   buffer_data = 0;
        MemoryData     memory_data = 0;
        VkBuffer       buffer      = fixture.CreateBuffer(4096, row.usage, &buffer_data);
        VkDeviceMemory memory      = fixture.AllocateMemory(4096, row.capture_memory_type, &memory_data);

        // CHECK rather than REQUIRE, so a failing row does not hide the rows after it.
        VkMemoryPropertyFlags bind_properties = 0;
        const VkResult        result =
            fixture.allocator().BindBufferMemory(buffer, memory, 0, buffer_data, memory_data, &bind_properties);
        CHECK(result == VK_SUCCESS);
        if (result != VK_SUCCESS)
        {
            continue;
        }

        const BindRecord bind = fixture.fake().BindOf(VK_HANDLE_TO_UINT64(buffer));
        CHECK(fixture.fake().AllocationOf(bind.memory).memory_type == row.expected_replay_type);
    }
}

TEST_CASE("Rebind ignores the AMD device coherent bit when picking the image memory usage", "[rebind]")
{
    RebindFixture fixture;

    ResourceData   image_data  = 0;
    MemoryData     memory_data = 0;
    VkImage        image  = fixture.CreateImage(VK_IMAGE_TILING_LINEAR, VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &image_data);
    VkDeviceMemory memory = fixture.AllocateMemory(kImageSize, kCaptureDeviceLocalAmd, &memory_data);

    VkMemoryPropertyFlags bind_properties = 0;
    REQUIRE(fixture.allocator().BindImageMemory(image, memory, 0, image_data, memory_data, &bind_properties) ==
            VK_SUCCESS);

    // Device local without the AMD bit means GPU_ONLY. With the bit kept, it would be CPU_ONLY.
    const BindRecord bind = fixture.fake().BindOf(VK_HANDLE_TO_UINT64(image));
    CHECK(fixture.fake().AllocationOf(bind.memory).memory_type == kReplayGpuOnly);
}

TEST_CASE("Rebind gives an image a dedicated allocation when the device requires it", "[rebind]")
{
    SECTION("dedicated allocation extensions enabled")
    {
        RebindFixture fixture(
            { VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME });
        fixture.fake().image_requires_dedicated = true;

        ResourceData   image_data  = 0;
        MemoryData     memory_data = 0;
        VkImage        image  = fixture.CreateImage(VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT, &image_data);
        VkDeviceMemory memory = fixture.AllocateMemory(kImageSize, kCaptureDeviceLocal, &memory_data);

        VkMemoryPropertyFlags bind_properties = 0;
        REQUIRE(fixture.allocator().BindImageMemory(image, memory, 0, image_data, memory_data, &bind_properties) ==
                VK_SUCCESS);

        const BindRecord bind = fixture.fake().BindOf(VK_HANDLE_TO_UINT64(image));
        CHECK(fixture.fake().AllocationOf(bind.memory).dedicated_image == image);
        CHECK(bind.offset == 0);
    }

    SECTION("dedicated allocation extensions not enabled")
    {
        RebindFixture fixture;
        fixture.fake().image_requires_dedicated = true;

        ResourceData   image_data  = 0;
        MemoryData     memory_data = 0;
        VkImage        image  = fixture.CreateImage(VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT, &image_data);
        VkDeviceMemory memory = fixture.AllocateMemory(kImageSize, kCaptureDeviceLocal, &memory_data);

        VkMemoryPropertyFlags bind_properties = 0;
        REQUIRE(fixture.allocator().BindImageMemory(image, memory, 0, image_data, memory_data, &bind_properties) ==
                VK_SUCCESS);

        const BindRecord bind = fixture.fake().BindOf(VK_HANDLE_TO_UINT64(image));
        CHECK(fixture.fake().AllocationOf(bind.memory).dedicated_image == VK_NULL_HANDLE);
    }
}

TEST_CASE("Rebind suballocates disjoint resources from one device memory", "[rebind]")
{
    RebindFixture fixture;

    const VkBufferUsageFlags usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ResourceData             first_data  = 0;
    ResourceData             second_data = 0;
    MemoryData               memory_data = 0;
    VkBuffer                 first       = fixture.CreateBuffer(4096, usage, &first_data);
    VkBuffer                 second      = fixture.CreateBuffer(4096, usage, &second_data);
    VkDeviceMemory           memory      = fixture.AllocateMemory(8192, kCaptureDeviceLocal, &memory_data);

    VkMemoryPropertyFlags bind_properties = 0;
    REQUIRE(fixture.allocator().BindBufferMemory(first, memory, 0, first_data, memory_data, &bind_properties) ==
            VK_SUCCESS);
    REQUIRE(fixture.allocator().BindBufferMemory(second, memory, 4096, second_data, memory_data, &bind_properties) ==
            VK_SUCCESS);

    const BindRecord first_bind  = fixture.fake().BindOf(VK_HANDLE_TO_UINT64(first));
    const BindRecord second_bind = fixture.fake().BindOf(VK_HANDLE_TO_UINT64(second));

    CHECK(fixture.fake().allocations.size() == 1);
    CHECK(first_bind.memory == second_bind.memory);
    CHECK(first_bind.offset % kBufferAlignment == 0);
    CHECK(second_bind.offset % kBufferAlignment == 0);
    CHECK(((first_bind.offset + 4096 <= second_bind.offset) || (second_bind.offset + 4096 <= first_bind.offset)));
}

TEST_CASE("Rebind binds an aliased sub-range inside the aliased resource's allocation", "[rebind]")
{
    RebindFixture fixture;

    // Different usage, so the second buffer would pick a different memory type if it were not aliased.
    ResourceData outer_data  = 0;
    ResourceData inner_data  = 0;
    MemoryData   memory_data = 0;
    VkBuffer     outer =
        fixture.CreateBuffer(4096, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, &outer_data);
    VkBuffer       inner  = fixture.CreateBuffer(1024, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &inner_data);
    VkDeviceMemory memory = fixture.AllocateMemory(8192, kCaptureHostCoherent, &memory_data);

    VkMemoryPropertyFlags bind_properties = 0;
    REQUIRE(fixture.allocator().BindBufferMemory(outer, memory, 256, outer_data, memory_data, &bind_properties) ==
            VK_SUCCESS);
    REQUIRE(fixture.allocator().BindBufferMemory(inner, memory, 512, inner_data, memory_data, &bind_properties) ==
            VK_SUCCESS);

    const BindRecord outer_bind = fixture.fake().BindOf(VK_HANDLE_TO_UINT64(outer));
    const BindRecord inner_bind = fixture.fake().BindOf(VK_HANDLE_TO_UINT64(inner));

    CHECK(inner_bind.memory == outer_bind.memory);
    CHECK(inner_bind.offset == outer_bind.offset + 256);
}

TEST_CASE("Rebind frees every device memory it allocated", "[rebind]")
{
    RebindFixture fixture(
        { VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME });
    fixture.fake().image_requires_dedicated = true;

    ResourceData   image_data         = 0;
    ResourceData   buffer_data        = 0;
    MemoryData     image_memory_data  = 0;
    MemoryData     buffer_memory_data = 0;
    VkImage        image        = fixture.CreateImage(VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_SAMPLED_BIT, &image_data);
    VkBuffer       buffer       = fixture.CreateBuffer(4096, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, &buffer_data);
    VkDeviceMemory image_memory = fixture.AllocateMemory(kImageSize, kCaptureDeviceLocal, &image_memory_data);
    VkDeviceMemory buffer_memory = fixture.AllocateMemory(4096, kCaptureHostCoherent, &buffer_memory_data);

    VkMemoryPropertyFlags bind_properties = 0;
    REQUIRE(fixture.allocator().BindImageMemory(
                image, image_memory, 0, image_data, image_memory_data, &bind_properties) == VK_SUCCESS);
    REQUIRE(fixture.allocator().BindBufferMemory(
                buffer, buffer_memory, 0, buffer_data, buffer_memory_data, &bind_properties) == VK_SUCCESS);

    fixture.Teardown();

    const FakeDevice& fake = fixture.fake();
    REQUIRE(fake.allocations.size() == 2);
    CHECK(fake.frees.size() == fake.allocations.size());
    for (const AllocateRecord& allocation : fake.allocations)
    {
        CHECK(std::count(fake.frees.begin(), fake.frees.end(), allocation.memory) == 1);
    }
    CHECK(fake.destroyed_pools == fake.created_pools);
}

namespace
{

// Larger than half of VMA's preferred block size, so VMA gives each object its own VkDeviceMemory.
constexpr VkDeviceSize kDedicatedSessionMemorySize = VkDeviceSize{ 32 } << 20;

VkDataGraphPipelineSessionBindPointRequirementARM MemoryBindPoint(VkDataGraphPipelineSessionBindPointARM bind_point,
                                                                  uint32_t                               num_objects)
{
    VkDataGraphPipelineSessionBindPointRequirementARM requirement{
        VK_STRUCTURE_TYPE_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_REQUIREMENT_ARM
    };
    requirement.bindPoint     = bind_point;
    requirement.bindPointType = VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TYPE_MEMORY_ARM;
    requirement.numObjects    = num_objects;
    return requirement;
}

} // namespace

TEST_CASE("Rebind binds memory for every data graph session object in one call", "[rebind]")
{
    RebindFixture fixture;
    FakeDevice&   fake       = fixture.fake();
    fake.session_bind_points = { MemoryBindPoint(VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_OPTICAL_FLOW_CACHE_ARM, 2),
                                 MemoryBindPoint(VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM, 1) };

    VkDataGraphPipelineSessionARM session      = VK_NULL_HANDLE;
    ResourceData                  session_data = 0;
    REQUIRE(fixture.CreateSession(0, &session, &session_data) == VK_SUCCESS);

    REQUIRE(fake.session_bind_calls == 1);
    REQUIRE(fake.session_binds.size() == 3);
    REQUIRE(fake.session_memory_queries.size() == 3);

    const std::pair<VkDataGraphPipelineSessionBindPointARM, uint32_t> expected[] = {
        { VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_OPTICAL_FLOW_CACHE_ARM, 0 },
        { VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_OPTICAL_FLOW_CACHE_ARM, 1 },
        { VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM, 0 },
    };
    for (size_t i = 0; i < fake.session_binds.size(); ++i)
    {
        const auto& bind  = fake.session_binds[i];
        const auto& query = fake.session_memory_queries[i];
        CAPTURE(i);
        CHECK(bind.session == session);
        CHECK(bind.bindPoint == expected[i].first);
        CHECK(bind.objectIndex == expected[i].second);
        CHECK(query.session == session);
        CHECK(query.bindPoint == expected[i].first);
        CHECK(query.objectIndex == expected[i].second);
        CHECK(bind.memoryOffset % kBufferAlignment == 0);
        CHECK(fake.AllocationOf(bind.memory).memory_type == kReplayGpuOnly);
    }

    // Each object gets its own range, even where VMA places them in one VkDeviceMemory.
    for (size_t i = 0; i < fake.session_binds.size(); ++i)
    {
        for (size_t j = i + 1; j < fake.session_binds.size(); ++j)
        {
            const auto& a = fake.session_binds[i];
            const auto& b = fake.session_binds[j];
            CAPTURE(i, j);
            CHECK(((a.memory != b.memory) || (a.memoryOffset + fake.session_memory_size <= b.memoryOffset) ||
                   (b.memoryOffset + fake.session_memory_size <= a.memoryOffset)));
        }
    }
}

TEST_CASE("Rebind frees data graph session memory when the session is destroyed", "[rebind]")
{
    RebindFixture fixture;
    FakeDevice&   fake       = fixture.fake();
    fake.session_bind_points = { MemoryBindPoint(VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM, 2) };
    fake.session_memory_size = kDedicatedSessionMemorySize;

    VkDataGraphPipelineSessionARM session      = VK_NULL_HANDLE;
    ResourceData                  session_data = 0;
    REQUIRE(fixture.CreateSession(0, &session, &session_data) == VK_SUCCESS);
    REQUIRE(fake.allocations.size() == 2);
    REQUIRE(fake.frees.empty());

    fixture.DestroySession(session, session_data);

    CHECK(fake.destroyed_sessions == std::vector<VkDataGraphPipelineSessionARM>{ session });
    CHECK(fake.frees.size() == 2);
    for (const AllocateRecord& allocation : fake.allocations)
    {
        CHECK(std::count(fake.frees.begin(), fake.frees.end(), allocation.memory) == 1);
    }
}

TEST_CASE("Rebind puts data graph session memory in device local memory", "[rebind]")
{
    RebindFixture fixture;
    FakeDevice&   fake       = fixture.fake();
    fake.session_bind_points = { MemoryBindPoint(VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM, 1) };

    // The host visible type has the lower index, so VMA would take it without a preference.
    fake.memory_type_bits = (1u << kReplayCpuOnly) | (1u << kReplayGpuOnly);

    VkDataGraphPipelineSessionARM session      = VK_NULL_HANDLE;
    ResourceData                  session_data = 0;
    REQUIRE(fixture.CreateSession(0, &session, &session_data) == VK_SUCCESS);

    REQUIRE(fake.session_binds.size() == 1);
    CHECK(fake.AllocationOf(fake.session_binds[0].memory).memory_type == kReplayGpuOnly);
}

TEST_CASE("Rebind matches the protection of data graph session memory to the session", "[rebind]")
{
    RebindFixture fixture;
    FakeDevice&   fake       = fixture.fake();
    fake.session_bind_points = { MemoryBindPoint(VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM, 1) };

    VkDataGraphPipelineSessionARM session      = VK_NULL_HANDLE;
    ResourceData                  session_data = 0;

    SECTION("unprotected session avoids the protected type")
    {
        // The protected type is device local, so the GPU_ONLY preference would pick it.
        fake.memory_type_bits = (1u << kReplayCpuOnly) | (1u << kReplayProtected);
        REQUIRE(fixture.CreateSession(0, &session, &session_data) == VK_SUCCESS);

        REQUIRE(fake.session_binds.size() == 1);
        CHECK(fake.AllocationOf(fake.session_binds[0].memory).memory_type == kReplayCpuOnly);
    }

    SECTION("protected session uses the protected type")
    {
        fake.memory_type_bits = kAllUnprotectedTypes | (1u << kReplayProtected);
        REQUIRE(fixture.CreateSession(
                    VK_DATA_GRAPH_PIPELINE_SESSION_CREATE_PROTECTED_BIT_ARM, &session, &session_data) == VK_SUCCESS);

        REQUIRE(fake.session_binds.size() == 1);
        CHECK(fake.AllocationOf(fake.session_binds[0].memory).memory_type == kReplayProtected);
    }
}

TEST_CASE("Rebind releases everything when binding data graph session memory fails", "[rebind]")
{
    RebindFixture fixture;
    FakeDevice&   fake       = fixture.fake();
    fake.session_bind_points = { MemoryBindPoint(VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM, 2) };
    fake.session_memory_size = kDedicatedSessionMemorySize;
    fake.session_bind_result = VK_ERROR_OUT_OF_DEVICE_MEMORY;

    // Rebind logs this failure as an error. The line in the test output is expected.

    VkDataGraphPipelineSessionARM session      = VK_NULL_HANDLE;
    ResourceData                  session_data = 0;
    CHECK(fixture.CreateSession(0, &session, &session_data) == VK_ERROR_OUT_OF_DEVICE_MEMORY);

    CHECK(session == VK_NULL_HANDLE);
    CHECK(session_data == 0);
    CHECK(fake.destroyed_sessions == fake.created_sessions);
    REQUIRE(fake.allocations.size() == 2);
    CHECK(fake.frees.size() == 2);
    for (const AllocateRecord& allocation : fake.allocations)
    {
        CHECK(std::count(fake.frees.begin(), fake.frees.end(), allocation.memory) == 1);
    }
}

TEST_CASE("Rebind ignores captured data graph session memory binds", "[rebind]")
{
    RebindFixture fixture;
    FakeDevice&   fake       = fixture.fake();
    fake.session_bind_points = { MemoryBindPoint(VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM, 1) };

    VkDataGraphPipelineSessionARM session      = VK_NULL_HANDLE;
    ResourceData                  session_data = 0;
    REQUIRE(fixture.CreateSession(0, &session, &session_data) == VK_SUCCESS);

    MemoryData     memory_data = 0;
    VkDeviceMemory memory      = fixture.AllocateMemory(4096, kCaptureDeviceLocal, &memory_data);

    VkBindDataGraphPipelineSessionMemoryInfoARM captured_bind{
        VK_STRUCTURE_TYPE_BIND_DATA_GRAPH_PIPELINE_SESSION_MEMORY_INFO_ARM
    };
    captured_bind.session     = session;
    captured_bind.bindPoint   = VK_DATA_GRAPH_PIPELINE_SESSION_BIND_POINT_TRANSIENT_ARM;
    captured_bind.objectIndex = 0;
    captured_bind.memory      = memory;

    const size_t                    allocation_count = fake.allocations.size();
    constexpr VkMemoryPropertyFlags kUntouched       = 0x1234;
    VkMemoryPropertyFlags           bind_properties  = kUntouched;
    CHECK(fixture.allocator().BindDataGraphPipelineSessionMemory(
              1, &captured_bind, &session_data, &memory_data, &bind_properties) == VK_SUCCESS);
    CHECK(fixture.allocator().BindDataGraphPipelineSessionMemory(99, nullptr, nullptr, nullptr, nullptr) == VK_SUCCESS);

    CHECK(bind_properties == kUntouched);
    CHECK(fake.session_bind_calls == 1);
    CHECK(fake.allocations.size() == allocation_count);
}
