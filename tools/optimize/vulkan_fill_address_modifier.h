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

#ifndef GFXRECON_VULKAN_FILL_ADDRESS_MODIFIER_H
#define GFXRECON_VULKAN_FILL_ADDRESS_MODIFIER_H

#include "vulkan_modifier_base.h"

#include "util/spirv_parsing_util.h"

#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)

/**
 * @brief   The block that last wrote each byte of one device memory, by offset from the start of the memory.
 */
class FillWriterMap
{
  public:
    struct Writer
    {
        uint64_t block_index{ 0 };
        bool     is_fill2{ false }; ///< false for a FillMemoryCommand.
        uint64_t map_offset{ 0 };   ///< Map offset when the block was written, to turn memory offsets into mapped ones.

        bool operator==(const Writer& other) const = default;
    };

    enum class Coverage
    {
        kOneBlock,
        kSeveralBlocks,
        kNotWritten ///< At least one byte no block wrote.
    };

    /// Records that writer wrote [begin, end). Later writes replace earlier ones.
    void Write(uint64_t begin, uint64_t end, const Writer& writer);

    /// Which blocks last wrote [begin, end). Sets writer only for kOneBlock.
    Coverage Find(uint64_t begin, uint64_t end, Writer* writer) const;

  private:
    struct Piece
    {
        uint64_t end{ 0 };
        Writer   writer;
    };

    std::map<uint64_t, Piece> pieces_; ///< By begin. Pieces do not overlap.
};

/**
 * @brief   Finds the device addresses in fill memory data, from the shaders that read them.
 *
 * The scan pass follows pipelines, descriptor sets, buffer binds and command buffers. At each submit
 * it turns the pointer fields that SPIR-V reflection reports in bound uniform and storage buffers
 * into locations in device memory, and gives each location to the fill block that last wrote it.
 *
 * Scope: single pointers in buffer descriptors. Chains, pointer arrays, push constants, inline
 * uniform blocks, shader binding tables and descriptor buffers are not followed.
 */
class VulkanFillAddressModifier : public VulkanModifierBase
{
  public:
    /// A pointer location in device memory, as memory id and offset from the start of the memory.
    using Location = std::pair<format::HandleId, uint64_t>;

    bool CanOptimize() override;

    void ProcessFillMemoryCommand(uint64_t memory_id, uint64_t offset, uint64_t size, const uint8_t* data) override;
    void ProcessFillMemoryCommand2(uint64_t                                     memory_id,
                                   const decode::FillMemoryModifiers&           modifiers,
                                   const std::vector<decode::FillMemoryRecord>& records) override;

    void Process_vkFreeMemory(const decode::ApiCallInfo& call_info, decode::args::FreeMemory& args) override;
    void Process_vkGetBufferDeviceAddress(const decode::ApiCallInfo&            call_info,
                                          decode::args::GetBufferDeviceAddress& args) override;
    void Process_vkGetBufferDeviceAddressKHR(const decode::ApiCallInfo&               call_info,
                                             decode::args::GetBufferDeviceAddressKHR& args) override;
    void Process_vkGetBufferDeviceAddressEXT(const decode::ApiCallInfo&               call_info,
                                             decode::args::GetBufferDeviceAddressEXT& args) override;

    void Process_vkCreateBuffer(const decode::ApiCallInfo& call_info, decode::args::CreateBuffer& args) override;
    void Process_vkBindBufferMemory(const decode::ApiCallInfo&      call_info,
                                    decode::args::BindBufferMemory& args) override;
    void Process_vkBindBufferMemory2(const decode::ApiCallInfo&       call_info,
                                     decode::args::BindBufferMemory2& args) override;
    void Process_vkBindBufferMemory2KHR(const decode::ApiCallInfo&          call_info,
                                        decode::args::BindBufferMemory2KHR& args) override;

    void Process_vkMapMemory(const decode::ApiCallInfo& call_info, decode::args::MapMemory& args) override;
    void Process_vkMapMemory2(const decode::ApiCallInfo& call_info, decode::args::MapMemory2& args) override;
    void Process_vkMapMemory2KHR(const decode::ApiCallInfo& call_info, decode::args::MapMemory2KHR& args) override;

    void Process_vkCreateShaderModule(const decode::ApiCallInfo&        call_info,
                                      decode::args::CreateShaderModule& args) override;
    void Process_vkCreateGraphicsPipelines(const decode::ApiCallInfo&             call_info,
                                           decode::args::CreateGraphicsPipelines& args) override;
    void Process_vkCreateComputePipelines(const decode::ApiCallInfo&            call_info,
                                          decode::args::CreateComputePipelines& args) override;
    void Process_vkCreateRayTracingPipelinesKHR(const decode::ApiCallInfo&                  call_info,
                                                decode::args::CreateRayTracingPipelinesKHR& args) override;

    void Process_vkCreateDescriptorSetLayout(const decode::ApiCallInfo&               call_info,
                                             decode::args::CreateDescriptorSetLayout& args) override;
    void Process_vkAllocateDescriptorSets(const decode::ApiCallInfo&            call_info,
                                          decode::args::AllocateDescriptorSets& args) override;
    void Process_vkFreeDescriptorSets(const decode::ApiCallInfo&        call_info,
                                      decode::args::FreeDescriptorSets& args) override;
    void Process_vkResetDescriptorPool(const decode::ApiCallInfo&         call_info,
                                       decode::args::ResetDescriptorPool& args) override;
    void Process_vkDestroyDescriptorPool(const decode::ApiCallInfo&           call_info,
                                         decode::args::DestroyDescriptorPool& args) override;
    void Process_vkUpdateDescriptorSets(const decode::ApiCallInfo&          call_info,
                                        decode::args::UpdateDescriptorSets& args) override;
    void Process_vkCreateDescriptorUpdateTemplate(const decode::ApiCallInfo&                    call_info,
                                                  decode::args::CreateDescriptorUpdateTemplate& args) override;
    void Process_vkCreateDescriptorUpdateTemplateKHR(const decode::ApiCallInfo&                       call_info,
                                                     decode::args::CreateDescriptorUpdateTemplateKHR& args) override;
    void Process_vkUpdateDescriptorSetWithTemplate(const decode::ApiCallInfo&                     call_info,
                                                   decode::args::UpdateDescriptorSetWithTemplate& args) override;
    void Process_vkUpdateDescriptorSetWithTemplateKHR(const decode::ApiCallInfo&                        call_info,
                                                      decode::args::UpdateDescriptorSetWithTemplateKHR& args) override;

    void Process_vkBeginCommandBuffer(const decode::ApiCallInfo&        call_info,
                                      decode::args::BeginCommandBuffer& args) override;
    void Process_vkFreeCommandBuffers(const decode::ApiCallInfo&        call_info,
                                      decode::args::FreeCommandBuffers& args) override;
    void Process_vkCmdBindPipeline(const decode::ApiCallInfo& call_info, decode::args::CmdBindPipeline& args) override;
    void Process_vkCmdBindDescriptorSets(const decode::ApiCallInfo&           call_info,
                                         decode::args::CmdBindDescriptorSets& args) override;
    void Process_vkCmdExecuteCommands(const decode::ApiCallInfo&        call_info,
                                      decode::args::CmdExecuteCommands& args) override;

    void Process_vkQueueSubmit(const decode::ApiCallInfo& call_info, decode::args::QueueSubmit& args) override;
    void Process_vkQueueSubmit2(const decode::ApiCallInfo& call_info, decode::args::QueueSubmit2& args) override;
    void Process_vkQueueSubmit2KHR(const decode::ApiCallInfo& call_info, decode::args::QueueSubmit2KHR& args) override;

// Calls that read bound descriptors: record a use for the bind point.
#define GFXRECON_FILL_ADDRESS_USE(call, bind_point)                                                \
    void Process_vk##call(const decode::ApiCallInfo& call_info, decode::args::call& args) override \
    {                                                                                              \
        GFXRECON_UNREFERENCED_PARAMETER(call_info);                                                \
        RecordUse(args.commandBuffer, bind_point);                                                 \
    }
    GFXRECON_FILL_ADDRESS_USE(CmdDraw, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawIndexed, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawIndirect, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawIndexedIndirect, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawIndirectCount, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawIndexedIndirectCount, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawIndirectCountKHR, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawIndexedIndirectCountKHR, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawIndirectCountAMD, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawIndexedIndirectCountAMD, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawIndirectByteCountEXT, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawMultiEXT, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawMultiIndexedEXT, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawMeshTasksEXT, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawMeshTasksIndirectEXT, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDrawMeshTasksIndirectCountEXT, VK_PIPELINE_BIND_POINT_GRAPHICS)
    GFXRECON_FILL_ADDRESS_USE(CmdDispatch, VK_PIPELINE_BIND_POINT_COMPUTE)
    GFXRECON_FILL_ADDRESS_USE(CmdDispatchIndirect, VK_PIPELINE_BIND_POINT_COMPUTE)
    GFXRECON_FILL_ADDRESS_USE(CmdDispatchBase, VK_PIPELINE_BIND_POINT_COMPUTE)
    GFXRECON_FILL_ADDRESS_USE(CmdDispatchBaseKHR, VK_PIPELINE_BIND_POINT_COMPUTE)
    GFXRECON_FILL_ADDRESS_USE(CmdTraceRaysKHR, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
    GFXRECON_FILL_ADDRESS_USE(CmdTraceRaysIndirectKHR, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
    GFXRECON_FILL_ADDRESS_USE(CmdTraceRaysIndirect2KHR, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR)
#undef GFXRECON_FILL_ADDRESS_USE

// Calls that set state this pass does not follow: count them, so the summary shows what it missed.
#define GFXRECON_FILL_ADDRESS_NOT_FOLLOWED(call)                                                   \
    void Process_vk##call(const decode::ApiCallInfo& call_info, decode::args::call& args) override \
    {                                                                                              \
        GFXRECON_UNREFERENCED_PARAMETER(call_info);                                                \
        GFXRECON_UNREFERENCED_PARAMETER(args);                                                     \
        ++not_followed_calls_["vk" #call];                                                         \
    }
    GFXRECON_FILL_ADDRESS_NOT_FOLLOWED(CmdBindDescriptorSets2)
    GFXRECON_FILL_ADDRESS_NOT_FOLLOWED(CmdBindDescriptorSets2KHR)
    GFXRECON_FILL_ADDRESS_NOT_FOLLOWED(CmdPushDescriptorSet)
    GFXRECON_FILL_ADDRESS_NOT_FOLLOWED(CmdPushDescriptorSetKHR)
    GFXRECON_FILL_ADDRESS_NOT_FOLLOWED(CmdPushDescriptorSetWithTemplate)
    GFXRECON_FILL_ADDRESS_NOT_FOLLOWED(CmdPushDescriptorSetWithTemplateKHR)
    GFXRECON_FILL_ADDRESS_NOT_FOLLOWED(CmdSetDescriptorBufferOffsetsEXT)
    GFXRECON_FILL_ADDRESS_NOT_FOLLOWED(CreateShadersEXT)
#undef GFXRECON_FILL_ADDRESS_NOT_FOLLOWED

  private:
    using BufferReferenceInfo = util::SpirVParsingUtil::BufferReferenceInfo;
    using References          = std::set<BufferReferenceInfo>;

    struct TrackedBuffer
    {
        uint64_t         size{ 0 };
        format::HandleId memory_id{ format::kNullHandleId };
        uint64_t         memory_offset{ 0 };
    };

    struct LayoutBinding
    {
        VkDescriptorType type{ VK_DESCRIPTOR_TYPE_MAX_ENUM };
        uint32_t         count{ 0 };
    };
    using SetLayout = std::map<uint32_t, LayoutBinding>;

    struct BufferDescriptor
    {
        format::HandleId buffer_id{ format::kNullHandleId };
        uint64_t         offset{ 0 };
        uint64_t         range{ 0 };
    };

    struct TrackedSet
    {
        format::HandleId                                  pool_id{ format::kNullHandleId };
        const SetLayout*                                  layout{ nullptr };
        std::map<uint32_t, std::vector<BufferDescriptor>> bindings; ///< Buffer descriptors only.
    };

    /// One descriptor update template entry for buffer descriptors, in the order the capture packs them.
    struct TemplateEntry
    {
        uint32_t binding{ 0 };
        uint32_t array_element{ 0 };
        uint32_t count{ 0 };
    };

    struct BoundSet
    {
        format::HandleId      set_id{ format::kNullHandleId };
        std::vector<uint32_t> dynamic_offsets;

        bool operator==(const BoundSet& other) const = default;
    };

    /// The state a draw, dispatch or trace reads its descriptors from.
    struct Use
    {
        format::HandleId             pipeline_id{ format::kNullHandleId };
        std::map<uint32_t, BoundSet> sets;

        bool operator==(const Use& other) const = default;
    };

    struct TrackedCommandBuffer
    {
        std::map<VkPipelineBindPoint, Use> bound;
        std::vector<Use>                   uses;
        std::vector<format::HandleId>      secondaries;
    };

    template <typename Args>
    void RecordBufferBinds2(Args& args);
    template <typename Args>
    void RecordTemplate(Args& args);
    template <typename Args>
    void RecordTemplateUpdate(Args& args);
    template <typename Args>
    void RecordSubmit2(Args& args);
    template <typename Args>
    void RecordPipelines(Args& args);
    template <typename Args>
    void RecordDeviceAddress(Args& args);

    void RecordMap(format::HandleId memory_id, uint64_t offset);
    void RecordFill(format::HandleId memory_id, uint64_t offset, uint64_t size, const uint8_t* data, bool is_fill2);
    void AssignWriter(const Location& location);
    void CountValue(const Location& location);
    void AddStageReferences(format::HandleId                                       pipeline_id,
                            const decode::Decoded_VkPipelineShaderStageCreateInfo& stage_meta,
                            const VkPipelineShaderStageCreateInfo&                 stage);
    void AddLibraryReferences(format::HandleId                                      pipeline_id,
                              const decode::Decoded_VkPipelineLibraryCreateInfoKHR* library);
    void WriteBuffers(format::HandleId                              set_id,
                      uint32_t                                      binding,
                      uint32_t                                      array_element,
                      uint32_t                                      count,
                      const decode::Decoded_VkDescriptorBufferInfo* buffer_infos);
    void RecordUse(format::HandleId command_buffer_id, VkPipelineBindPoint bind_point);
    void ResolveCommandBuffer(format::HandleId command_buffer_id, uint32_t depth);
    void ResolveUse(const Use& use);

    std::unordered_map<format::HandleId, TrackedBuffer>              buffers_;
    std::unordered_map<format::HandleId, uint64_t>                   map_offsets_;
    std::unordered_map<format::HandleId, References>                 module_references_;
    std::unordered_map<format::HandleId, References>                 pipeline_references_;
    std::unordered_map<format::HandleId, SetLayout>                  set_layouts_;
    std::unordered_map<format::HandleId, TrackedSet>                 sets_;
    std::unordered_map<format::HandleId, std::vector<TemplateEntry>> templates_;
    std::unordered_map<format::HandleId, TrackedCommandBuffer>       command_buffers_;

    /// Pointer locations read by submitted work, with the number of submits that read each.
    std::map<Location, uint64_t> locations_;

    std::unordered_map<format::HandleId, FillWriterMap>        writers_;
    std::unordered_map<format::HandleId, std::vector<uint8_t>> fill_data_; ///< Last fill data, by memory offset.
    std::map<VkDeviceAddress, VkDeviceAddress> capture_addresses_;         ///< Buffer address ranges, begin to end.

    /// Pointer locations per FillMemoryCommand2 block, as offsets from the mapped pointer.
    std::map<uint64_t, std::set<uint64_t>> block_values_;

    std::map<std::string, uint64_t> location_reads_; ///< Locations read at submit, by who last wrote them.
    std::map<std::string, uint64_t> values_;         ///< Values at the locations given to a block, by kind.

    uint64_t                        submitted_uses_{ 0 };
    std::map<std::string, uint64_t> skipped_;            ///< Pointer fields not turned into a location, by reason.
    std::map<std::string, uint64_t> not_followed_calls_; ///< Calls whose state this pass ignores.
};

GFXRECON_END_NAMESPACE(gfxrecon)

#endif // GFXRECON_VULKAN_FILL_ADDRESS_MODIFIER_H
