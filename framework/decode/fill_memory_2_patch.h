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

#ifndef GFXRECON_DECODE_FILL_MEMORY_2_PATCH_H
#define GFXRECON_DECODE_FILL_MEMORY_2_PATCH_H

#include "decode/metadata_consumer_base.h"
#include "decode/vulkan_device_address_tracker.h"
#include "format/format.h"
#include "util/defines.h"

#include <cstdint>
#include <functional>
#include <vector>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)
GFXRECON_BEGIN_NAMESPACE(decode)

// Size of a value of this type in bytes, or 0 when the caller cannot map it.
using ResourceValueSizeFunction = std::function<uint32_t(format::ResourceValueType type)>;

// Maps one value in place.
using ResourceValueMapFunction = std::function<void(format::ResourceValueType type, uint8_t* value, uint32_t size)>;

/// @brief Copies the record data into storage, maps each resource value in the copy, and points the records at it.
///
/// A value may span records, but must lie fully inside them. A value that does not, or whose size is 0, is skipped.
/// @return false if a value was skipped.
bool PatchFillMemoryRecords(const std::vector<FillMemoryResourceValue>& values,
                            const ResourceValueSizeFunction&            get_size,
                            const ResourceValueMapFunction&             map_value,
                            std::vector<uint8_t>*                       storage,
                            std::vector<FillMemoryRecord>*              records);

// Size of a Vulkan resource value that replay can map, or 0.
uint32_t GetVulkanResourceValueSize(format::ResourceValueType type);

// Maps a capture-time device address to its replay-time address. Leaves null and unknown addresses as they are.
void MapVulkanResourceValue(const VulkanDeviceAddressTracker& address_tracker,
                            format::ResourceValueType         type,
                            uint8_t*                          value,
                            uint32_t                          size);

GFXRECON_END_NAMESPACE(decode)
GFXRECON_END_NAMESPACE(gfxrecon)

#endif // GFXRECON_DECODE_FILL_MEMORY_2_PATCH_H
