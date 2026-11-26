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

#include "vulkan_fill_address_modifier.h"

#include "decode/vulkan_pnext_node.h"
#include "graphics/vulkan_struct_get_pnext.h"
#include "util/logging.h"

#include <iterator>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)

namespace
{

using References = std::set<util::SpirVParsingUtil::BufferReferenceInfo>;

void Reflect(const uint32_t* code, size_t code_size, References* references)
{
    util::SpirVParsingUtil parser;
    if ((code != nullptr) && parser.ParseBufferReferences(code, code_size))
    {
        const auto found = parser.GetBufferReferenceInfos();
        references->insert(found.begin(), found.end());
    }
}

bool IsBufferDescriptor(VkDescriptorType type)
{
    return (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) || (type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) ||
           (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) || (type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
}

bool IsDynamic(VkDescriptorType type)
{
    return (type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC) || (type == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC);
}

// Moves binding and element past the end of a binding, as consecutive descriptor updates do.
// Returns false at the end of the layout.
template <typename Layout>
bool NormalizeBinding(const Layout& layout, uint32_t* binding, uint32_t* element)
{
    auto entry = layout.find(*binding);
    while ((entry != layout.end()) && (*element >= entry->second.count))
    {
        *element -= entry->second.count;
        ++entry;
        if (entry != layout.end())
        {
            *binding = entry->first;
        }
    }
    return entry != layout.end();
}

} // namespace

bool VulkanFillAddressModifier::CanOptimize()
{
    std::map<format::HandleId, uint64_t> per_memory;
    for (const auto& [location, submits] : locations_)
    {
        ++per_memory[location.first];
    }

    GFXRECON_LOG_INFO("Fill address scan: %" PRIu64 " submitted uses, %zu pointer locations in %zu memories.",
                      submitted_uses_,
                      locations_.size(),
                      per_memory.size());
    for (const auto& [memory_id, count] : per_memory)
    {
        GFXRECON_LOG_INFO("Fill address scan:   memory %" PRIu64 ": %" PRIu64 " locations", memory_id, count);
    }
    for (const auto& [reason, count] : skipped_)
    {
        GFXRECON_LOG_INFO("Fill address scan: skipped %" PRIu64 " pointer fields: %s", count, reason.c_str());
    }
    for (const auto& [call, count] : not_followed_calls_)
    {
        GFXRECON_LOG_INFO("Fill address scan: %" PRIu64 " calls to %s not followed", count, call.c_str());
    }

    // The write pass does not exist yet.
    return false;
}

void VulkanFillAddressModifier::Process_vkCreateBuffer(const decode::ApiCallInfo&  call_info,
                                                       decode::args::CreateBuffer& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (IsModificationPass() || args.pBuffer.IsNull() || args.pCreateInfo.IsNull())
    {
        return;
    }
    buffers_[*args.pBuffer.GetPointer()].size = args.pCreateInfo.GetPointer()->size;
}

void VulkanFillAddressModifier::Process_vkBindBufferMemory(const decode::ApiCallInfo&      call_info,
                                                           decode::args::BindBufferMemory& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (IsModificationPass())
    {
        return;
    }
    auto& buffer         = buffers_[args.buffer];
    buffer.memory_id     = args.memory;
    buffer.memory_offset = args.memoryOffset;
}

template <typename Args>
void VulkanFillAddressModifier::RecordBufferBinds2(Args& args)
{
    if (IsModificationPass() || args.pBindInfos.IsNull())
    {
        return;
    }
    const auto* bind_infos = args.pBindInfos.GetMetaStructPointer();
    for (uint32_t i = 0; i < args.bindInfoCount; ++i)
    {
        auto& buffer         = buffers_[bind_infos[i].buffer];
        buffer.memory_id     = bind_infos[i].memory;
        buffer.memory_offset = bind_infos[i].decoded_value->memoryOffset;
    }
}

void VulkanFillAddressModifier::Process_vkBindBufferMemory2(const decode::ApiCallInfo&       call_info,
                                                            decode::args::BindBufferMemory2& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    RecordBufferBinds2(args);
}

void VulkanFillAddressModifier::Process_vkBindBufferMemory2KHR(const decode::ApiCallInfo&          call_info,
                                                               decode::args::BindBufferMemory2KHR& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    RecordBufferBinds2(args);
}

void VulkanFillAddressModifier::RecordMap(format::HandleId memory_id, uint64_t offset)
{
    if (!IsModificationPass())
    {
        map_offsets_[memory_id] = offset;
    }
}

void VulkanFillAddressModifier::Process_vkMapMemory(const decode::ApiCallInfo& call_info, decode::args::MapMemory& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    RecordMap(args.memory, args.offset);
}

void VulkanFillAddressModifier::Process_vkMapMemory2(const decode::ApiCallInfo& call_info,
                                                     decode::args::MapMemory2&  args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (!args.pMemoryMapInfo.IsNull())
    {
        RecordMap(args.pMemoryMapInfo.GetMetaStructPointer()->memory, args.pMemoryMapInfo.GetPointer()->offset);
    }
}

void VulkanFillAddressModifier::Process_vkMapMemory2KHR(const decode::ApiCallInfo&   call_info,
                                                        decode::args::MapMemory2KHR& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (!args.pMemoryMapInfo.IsNull())
    {
        RecordMap(args.pMemoryMapInfo.GetMetaStructPointer()->memory, args.pMemoryMapInfo.GetPointer()->offset);
    }
}

void VulkanFillAddressModifier::Process_vkCreateShaderModule(const decode::ApiCallInfo&        call_info,
                                                             decode::args::CreateShaderModule& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (IsModificationPass() || args.pShaderModule.IsNull() || args.pCreateInfo.IsNull())
    {
        return;
    }
    const auto* create_info = args.pCreateInfo.GetPointer();
    Reflect(create_info->pCode, create_info->codeSize, &module_references_[*args.pShaderModule.GetPointer()]);
}

void VulkanFillAddressModifier::AddStageReferences(format::HandleId                                       pipeline_id,
                                                   const decode::Decoded_VkPipelineShaderStageCreateInfo& stage_meta,
                                                   const VkPipelineShaderStageCreateInfo&                 stage)
{
    auto& references = pipeline_references_[pipeline_id];
    if (stage_meta.module != format::kNullHandleId)
    {
        const auto entry = module_references_.find(stage_meta.module);
        if (entry != module_references_.end())
        {
            references.insert(entry->second.begin(), entry->second.end());
        }
    }
    else if (const auto* module = graphics::vulkan_struct_get_pnext<VkShaderModuleCreateInfo>(&stage))
    {
        // SPIR-V inlined in the stage (VK_KHR_maintenance5).
        Reflect(module->pCode, module->codeSize, &references);
    }
}

void VulkanFillAddressModifier::AddLibraryReferences(format::HandleId                                      pipeline_id,
                                                     const decode::Decoded_VkPipelineLibraryCreateInfoKHR* library)
{
    if ((library == nullptr) || library->pLibraries.IsNull())
    {
        return;
    }
    auto&       references  = pipeline_references_[pipeline_id];
    const auto* library_ids = library->pLibraries.GetPointer();
    for (size_t i = 0; i < library->pLibraries.GetLength(); ++i)
    {
        const auto entry = pipeline_references_.find(library_ids[i]);
        if (entry != pipeline_references_.end())
        {
            references.insert(entry->second.begin(), entry->second.end());
        }
    }
}

template <typename Args>
void VulkanFillAddressModifier::RecordPipelines(Args& args)
{
    if (IsModificationPass() || args.pPipelines.IsNull() || args.pCreateInfos.IsNull())
    {
        return;
    }
    const auto* pipeline_ids = args.pPipelines.GetPointer();
    const auto* infos_meta   = args.pCreateInfos.GetMetaStructPointer();
    const auto* infos        = args.pCreateInfos.GetPointer();

    for (size_t i = 0; i < args.pPipelines.GetLength(); ++i)
    {
        const auto& info_meta = infos_meta[i];
        if constexpr (std::is_same_v<Args, decode::args::CreateComputePipelines>)
        {
            AddStageReferences(pipeline_ids[i], *info_meta.stage, infos[i].stage);
        }
        else
        {
            if (info_meta.pStages != nullptr && !info_meta.pStages->IsNull())
            {
                const auto* stages_meta = info_meta.pStages->GetMetaStructPointer();
                for (size_t s = 0; s < info_meta.pStages->GetLength(); ++s)
                {
                    AddStageReferences(pipeline_ids[i], stages_meta[s], infos[i].pStages[s]);
                }
            }

            if constexpr (std::is_same_v<Args, decode::args::CreateRayTracingPipelinesKHR>)
            {
                if (info_meta.pLibraryInfo != nullptr && !info_meta.pLibraryInfo->IsNull())
                {
                    AddLibraryReferences(pipeline_ids[i], info_meta.pLibraryInfo->GetMetaStructPointer());
                }
            }
            else
            {
                AddLibraryReferences(
                    pipeline_ids[i],
                    decode::GetPNextMetaStruct<decode::Decoded_VkPipelineLibraryCreateInfoKHR>(info_meta.pNext));
            }
        }
    }
}

void VulkanFillAddressModifier::Process_vkCreateGraphicsPipelines(const decode::ApiCallInfo&             call_info,
                                                                  decode::args::CreateGraphicsPipelines& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    RecordPipelines(args);
}

void VulkanFillAddressModifier::Process_vkCreateComputePipelines(const decode::ApiCallInfo&            call_info,
                                                                 decode::args::CreateComputePipelines& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    RecordPipelines(args);
}

void VulkanFillAddressModifier::Process_vkCreateRayTracingPipelinesKHR(const decode::ApiCallInfo& call_info,
                                                                       decode::args::CreateRayTracingPipelinesKHR& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    RecordPipelines(args);
}

void VulkanFillAddressModifier::Process_vkCreateDescriptorSetLayout(const decode::ApiCallInfo&               call_info,
                                                                    decode::args::CreateDescriptorSetLayout& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (IsModificationPass() || args.pSetLayout.IsNull() || args.pCreateInfo.IsNull())
    {
        return;
    }
    const auto* create_info = args.pCreateInfo.GetPointer();
    auto&       layout      = set_layouts_[*args.pSetLayout.GetPointer()];
    for (uint32_t i = 0; i < create_info->bindingCount; ++i)
    {
        const auto& binding     = create_info->pBindings[i];
        layout[binding.binding] = { binding.descriptorType, binding.descriptorCount };
    }
}

void VulkanFillAddressModifier::Process_vkAllocateDescriptorSets(const decode::ApiCallInfo&            call_info,
                                                                 decode::args::AllocateDescriptorSets& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (IsModificationPass() || args.pDescriptorSets.IsNull() || args.pAllocateInfo.IsNull())
    {
        return;
    }
    const auto* allocate_info = args.pAllocateInfo.GetMetaStructPointer();
    const auto* layout_ids    = allocate_info->pSetLayouts.GetPointer();
    const auto* set_ids       = args.pDescriptorSets.GetPointer();

    for (size_t i = 0; i < args.pDescriptorSets.GetLength(); ++i)
    {
        const auto layout = set_layouts_.find(layout_ids[i]);
        sets_[set_ids[i]] = { allocate_info->descriptorPool,
                              (layout != set_layouts_.end()) ? &layout->second : nullptr,
                              {} };
    }
}

void VulkanFillAddressModifier::Process_vkFreeDescriptorSets(const decode::ApiCallInfo&        call_info,
                                                             decode::args::FreeDescriptorSets& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (IsModificationPass() || args.pDescriptorSets.IsNull())
    {
        return;
    }
    const auto* set_ids = args.pDescriptorSets.GetPointer();
    for (size_t i = 0; i < args.pDescriptorSets.GetLength(); ++i)
    {
        sets_.erase(set_ids[i]);
    }
}

void VulkanFillAddressModifier::Process_vkResetDescriptorPool(const decode::ApiCallInfo&         call_info,
                                                              decode::args::ResetDescriptorPool& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (!IsModificationPass())
    {
        std::erase_if(sets_, [&args](const auto& entry) { return entry.second.pool_id == args.descriptorPool; });
    }
}

void VulkanFillAddressModifier::Process_vkDestroyDescriptorPool(const decode::ApiCallInfo&           call_info,
                                                                decode::args::DestroyDescriptorPool& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (!IsModificationPass())
    {
        std::erase_if(sets_, [&args](const auto& entry) { return entry.second.pool_id == args.descriptorPool; });
    }
}

void VulkanFillAddressModifier::WriteBuffers(format::HandleId                              set_id,
                                             uint32_t                                      binding,
                                             uint32_t                                      array_element,
                                             uint32_t                                      count,
                                             const decode::Decoded_VkDescriptorBufferInfo* buffer_infos)
{
    const auto set = sets_.find(set_id);
    if ((set == sets_.end()) || (set->second.layout == nullptr) || (buffer_infos == nullptr))
    {
        ++skipped_["descriptor write to an unknown set or layout"];
        return;
    }
    const SetLayout& layout = *set->second.layout;

    for (uint32_t i = 0; i < count; ++i, ++array_element)
    {
        if (!NormalizeBinding(layout, &binding, &array_element))
        {
            break;
        }
        auto& descriptors = set->second.bindings[binding];
        descriptors.resize(layout.at(binding).count);
        descriptors[array_element] = { buffer_infos[i].buffer,
                                       buffer_infos[i].decoded_value->offset,
                                       buffer_infos[i].decoded_value->range };
    }
}

void VulkanFillAddressModifier::Process_vkUpdateDescriptorSets(const decode::ApiCallInfo&          call_info,
                                                               decode::args::UpdateDescriptorSets& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (IsModificationPass())
    {
        return;
    }

    if (!args.pDescriptorWrites.IsNull())
    {
        const auto* writes      = args.pDescriptorWrites.GetPointer();
        const auto* writes_meta = args.pDescriptorWrites.GetMetaStructPointer();
        for (uint32_t i = 0; i < args.descriptorWriteCount; ++i)
        {
            if (IsBufferDescriptor(writes[i].descriptorType) && (writes_meta[i].pBufferInfo != nullptr) &&
                !writes_meta[i].pBufferInfo->IsNull())
            {
                WriteBuffers(writes_meta[i].dstSet,
                             writes[i].dstBinding,
                             writes[i].dstArrayElement,
                             writes[i].descriptorCount,
                             writes_meta[i].pBufferInfo->GetMetaStructPointer());
            }
        }
    }

    if (!args.pDescriptorCopies.IsNull())
    {
        const auto* copies      = args.pDescriptorCopies.GetPointer();
        const auto* copies_meta = args.pDescriptorCopies.GetMetaStructPointer();
        for (uint32_t i = 0; i < args.descriptorCopyCount; ++i)
        {
            const auto src = sets_.find(copies_meta[i].srcSet);
            const auto dst = sets_.find(copies_meta[i].dstSet);
            if ((src == sets_.end()) || (dst == sets_.end()) || (src->second.layout == nullptr) ||
                (dst->second.layout == nullptr))
            {
                ++skipped_["descriptor copy with an unknown set or layout"];
                continue;
            }

            uint32_t src_binding = copies[i].srcBinding;
            uint32_t src_element = copies[i].srcArrayElement;
            uint32_t dst_binding = copies[i].dstBinding;
            uint32_t dst_element = copies[i].dstArrayElement;
            for (uint32_t c = 0; c < copies[i].descriptorCount; ++c, ++src_element, ++dst_element)
            {
                if (!NormalizeBinding(*src->second.layout, &src_binding, &src_element) ||
                    !NormalizeBinding(*dst->second.layout, &dst_binding, &dst_element))
                {
                    break;
                }

                // A copy of a descriptor this pass does not track clears the destination.
                BufferDescriptor descriptor;
                const auto       src_descriptors = src->second.bindings.find(src_binding);
                if ((src_descriptors != src->second.bindings.end()) && (src_element < src_descriptors->second.size()))
                {
                    descriptor = src_descriptors->second[src_element];
                }

                auto& dst_descriptors = dst->second.bindings[dst_binding];
                dst_descriptors.resize(dst->second.layout->at(dst_binding).count);
                dst_descriptors[dst_element] = descriptor;
            }
        }
    }
}

template <typename Args>
void VulkanFillAddressModifier::RecordTemplate(Args& args)
{
    if (IsModificationPass() || args.pDescriptorUpdateTemplate.IsNull() || args.pCreateInfo.IsNull())
    {
        return;
    }

    // The capture packs the buffer infos of all buffer entries together, in entry order.
    const auto* create_info = args.pCreateInfo.GetPointer();
    auto&       entries     = templates_[*args.pDescriptorUpdateTemplate.GetPointer()];
    for (uint32_t i = 0; i < create_info->descriptorUpdateEntryCount; ++i)
    {
        const auto& entry = create_info->pDescriptorUpdateEntries[i];
        if (IsBufferDescriptor(entry.descriptorType))
        {
            entries.push_back({ entry.dstBinding, entry.dstArrayElement, entry.descriptorCount });
        }
    }
}

void VulkanFillAddressModifier::Process_vkCreateDescriptorUpdateTemplate(
    const decode::ApiCallInfo& call_info, decode::args::CreateDescriptorUpdateTemplate& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    RecordTemplate(args);
}

void VulkanFillAddressModifier::Process_vkCreateDescriptorUpdateTemplateKHR(
    const decode::ApiCallInfo& call_info, decode::args::CreateDescriptorUpdateTemplateKHR& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    RecordTemplate(args);
}

template <typename Args>
void VulkanFillAddressModifier::RecordTemplateUpdate(Args& args)
{
    if (IsModificationPass())
    {
        return;
    }
    const auto entries = templates_.find(args.descriptorUpdateTemplate);
    if (entries == templates_.end())
    {
        ++skipped_["descriptor template update with an unknown template"];
        return;
    }

    const auto*  buffer_infos = args.pData.GetBufferInfoMetaStructPointer();
    const size_t buffer_count = args.pData.GetBufferInfoCount();
    size_t       first        = 0;
    for (const auto& entry : entries->second)
    {
        if (first + entry.count > buffer_count)
        {
            ++skipped_["descriptor template data shorter than its entries"];
            break;
        }
        WriteBuffers(args.descriptorSet, entry.binding, entry.array_element, entry.count, buffer_infos + first);
        first += entry.count;
    }
}

void VulkanFillAddressModifier::Process_vkUpdateDescriptorSetWithTemplate(
    const decode::ApiCallInfo& call_info, decode::args::UpdateDescriptorSetWithTemplate& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    RecordTemplateUpdate(args);
}

void VulkanFillAddressModifier::Process_vkUpdateDescriptorSetWithTemplateKHR(
    const decode::ApiCallInfo& call_info, decode::args::UpdateDescriptorSetWithTemplateKHR& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    RecordTemplateUpdate(args);
}

void VulkanFillAddressModifier::Process_vkBeginCommandBuffer(const decode::ApiCallInfo&        call_info,
                                                             decode::args::BeginCommandBuffer& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (!IsModificationPass())
    {
        command_buffers_[args.commandBuffer] = {};
    }
}

void VulkanFillAddressModifier::Process_vkFreeCommandBuffers(const decode::ApiCallInfo&        call_info,
                                                             decode::args::FreeCommandBuffers& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (IsModificationPass() || args.pCommandBuffers.IsNull())
    {
        return;
    }
    const auto* ids = args.pCommandBuffers.GetPointer();
    for (size_t i = 0; i < args.pCommandBuffers.GetLength(); ++i)
    {
        command_buffers_.erase(ids[i]);
    }
}

void VulkanFillAddressModifier::Process_vkCmdBindPipeline(const decode::ApiCallInfo&     call_info,
                                                          decode::args::CmdBindPipeline& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (!IsModificationPass())
    {
        command_buffers_[args.commandBuffer].bound[args.pipelineBindPoint].pipeline_id = args.pipeline;
    }
}

void VulkanFillAddressModifier::Process_vkCmdBindDescriptorSets(const decode::ApiCallInfo&           call_info,
                                                                decode::args::CmdBindDescriptorSets& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (IsModificationPass() || args.pDescriptorSets.IsNull())
    {
        return;
    }

    auto&           bound           = command_buffers_[args.commandBuffer].bound[args.pipelineBindPoint];
    const auto*     set_ids         = args.pDescriptorSets.GetPointer();
    const uint32_t* dynamic_offsets = args.pDynamicOffsets.IsNull() ? nullptr : args.pDynamicOffsets.GetPointer();
    uint32_t        next_dynamic    = 0;

    for (uint32_t i = 0; i < args.descriptorSetCount; ++i)
    {
        BoundSet bound_set{ set_ids[i], {} };

        // Dynamic offsets go to the dynamic descriptors of each set in binding order.
        const auto set = sets_.find(set_ids[i]);
        if ((set != sets_.end()) && (set->second.layout != nullptr))
        {
            for (const auto& [binding, layout_binding] : *set->second.layout)
            {
                for (uint32_t e = 0; IsDynamic(layout_binding.type) && (e < layout_binding.count); ++e)
                {
                    const bool has_offset = (dynamic_offsets != nullptr) && (next_dynamic < args.dynamicOffsetCount);
                    bound_set.dynamic_offsets.push_back(has_offset ? dynamic_offsets[next_dynamic] : 0);
                    ++next_dynamic;
                }
            }
        }
        bound.sets[args.firstSet + i] = std::move(bound_set);
    }
}

void VulkanFillAddressModifier::Process_vkCmdExecuteCommands(const decode::ApiCallInfo&        call_info,
                                                             decode::args::CmdExecuteCommands& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (IsModificationPass() || args.pCommandBuffers.IsNull())
    {
        return;
    }
    auto&       secondaries = command_buffers_[args.commandBuffer].secondaries;
    const auto* ids         = args.pCommandBuffers.GetPointer();
    secondaries.insert(secondaries.end(), ids, ids + args.pCommandBuffers.GetLength());
}

void VulkanFillAddressModifier::RecordUse(format::HandleId command_buffer_id, VkPipelineBindPoint bind_point)
{
    if (IsModificationPass())
    {
        return;
    }
    auto&      command_buffer = command_buffers_[command_buffer_id];
    const auto bound          = command_buffer.bound.find(bind_point);
    if ((bound == command_buffer.bound.end()) || !pipeline_references_.contains(bound->second.pipeline_id))
    {
        return;
    }

    // Descriptor contents are read at submit, so keep the bound sets, not the buffers.
    if (command_buffer.uses.empty() || !(command_buffer.uses.back() == bound->second))
    {
        command_buffer.uses.push_back(bound->second);
    }
}

void VulkanFillAddressModifier::ResolveUse(const Use& use)
{
    ++submitted_uses_;

    const auto references = pipeline_references_.find(use.pipeline_id);
    if (references == pipeline_references_.end())
    {
        return;
    }

    for (const auto& reference : references->second)
    {
        using Source = util::SpirVParsingUtil::BufferReferenceLocation;
        if ((reference.source != Source::UNIFORM_BUFFER) && (reference.source != Source::STORAGE_BUFFER))
        {
            ++skipped_["not in a buffer descriptor"];
            continue;
        }
        if (reference.chain)
        {
            ++skipped_["chain"];
            continue;
        }
        if (reference.array_stride != 0)
        {
            ++skipped_["pointer array"];
            continue;
        }

        const auto bound_set = use.sets.find(reference.set);
        if (bound_set == use.sets.end())
        {
            ++skipped_["set not bound"];
            continue;
        }
        const auto set = sets_.find(bound_set->second.set_id);
        if ((set == sets_.end()) || (set->second.layout == nullptr))
        {
            ++skipped_["unknown set"];
            continue;
        }
        const auto descriptors = set->second.bindings.find(reference.binding);
        if (descriptors == set->second.bindings.end())
        {
            ++skipped_["binding holds no buffer descriptor"];
            continue;
        }

        // Index of this binding's first dynamic offset in the set.
        uint32_t dynamic_index = 0;
        for (const auto& [binding, layout_binding] : *set->second.layout)
        {
            if (binding >= reference.binding)
            {
                break;
            }
            dynamic_index += IsDynamic(layout_binding.type) ? layout_binding.count : 0;
        }
        const bool dynamic = IsDynamic(set->second.layout->at(reference.binding).type);

        for (size_t element = 0; element < descriptors->second.size(); ++element)
        {
            const auto& descriptor = descriptors->second[element];
            if (descriptor.buffer_id == format::kNullHandleId)
            {
                continue;
            }
            const auto buffer = buffers_.find(descriptor.buffer_id);
            if ((buffer == buffers_.end()) || (buffer->second.memory_id == format::kNullHandleId))
            {
                ++skipped_["buffer not bound to memory"];
                continue;
            }

            uint64_t dynamic_offset = 0;
            if (dynamic)
            {
                const size_t index   = dynamic_index + element;
                const auto&  offsets = bound_set->second.dynamic_offsets;
                dynamic_offset       = (index < offsets.size()) ? offsets[index] : 0;
            }

            const uint64_t range =
                (descriptor.range == VK_WHOLE_SIZE) ? (buffer->second.size - descriptor.offset) : descriptor.range;
            if (reference.buffer_offset + sizeof(VkDeviceAddress) > range)
            {
                ++skipped_["outside the descriptor range"];
                continue;
            }

            const uint64_t offset =
                buffer->second.memory_offset + descriptor.offset + dynamic_offset + reference.buffer_offset;
            ++locations_[{ buffer->second.memory_id, offset }];
        }
    }
}

void VulkanFillAddressModifier::ResolveCommandBuffer(format::HandleId command_buffer_id, uint32_t depth)
{
    // Secondaries nest one level, or a few with VK_EXT_nested_command_buffer.
    constexpr uint32_t kMaxDepth = 8;
    const auto         entry     = command_buffers_.find(command_buffer_id);
    if ((entry == command_buffers_.end()) || (depth > kMaxDepth))
    {
        return;
    }
    for (const auto& use : entry->second.uses)
    {
        ResolveUse(use);
    }
    for (const auto secondary_id : entry->second.secondaries)
    {
        ResolveCommandBuffer(secondary_id, depth + 1);
    }
}

void VulkanFillAddressModifier::Process_vkQueueSubmit(const decode::ApiCallInfo& call_info,
                                                      decode::args::QueueSubmit& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    if (IsModificationPass() || args.pSubmits.IsNull())
    {
        return;
    }
    const auto* submits = args.pSubmits.GetMetaStructPointer();
    for (uint32_t i = 0; i < args.submitCount; ++i)
    {
        const auto* ids = submits[i].pCommandBuffers.GetPointer();
        for (size_t c = 0; c < submits[i].pCommandBuffers.GetLength(); ++c)
        {
            ResolveCommandBuffer(ids[c], 0);
        }
    }
}

template <typename Args>
void VulkanFillAddressModifier::RecordSubmit2(Args& args)
{
    if (IsModificationPass() || args.pSubmits.IsNull())
    {
        return;
    }
    const auto* submits = args.pSubmits.GetMetaStructPointer();
    for (uint32_t i = 0; i < args.submitCount; ++i)
    {
        if ((submits[i].pCommandBufferInfos == nullptr) || submits[i].pCommandBufferInfos->IsNull())
        {
            continue;
        }
        const auto* infos = submits[i].pCommandBufferInfos->GetMetaStructPointer();
        for (size_t c = 0; c < submits[i].pCommandBufferInfos->GetLength(); ++c)
        {
            ResolveCommandBuffer(infos[c].commandBuffer, 0);
        }
    }
}

void VulkanFillAddressModifier::Process_vkQueueSubmit2(const decode::ApiCallInfo&  call_info,
                                                       decode::args::QueueSubmit2& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    RecordSubmit2(args);
}

void VulkanFillAddressModifier::Process_vkQueueSubmit2KHR(const decode::ApiCallInfo&     call_info,
                                                          decode::args::QueueSubmit2KHR& args)
{
    GFXRECON_UNREFERENCED_PARAMETER(call_info);
    RecordSubmit2(args);
}

GFXRECON_END_NAMESPACE(gfxrecon)
