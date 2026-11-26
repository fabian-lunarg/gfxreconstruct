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

#include "decode/fill_memory_2_patch.h"

#include "util/platform.h"

#include <algorithm>
#include <limits>
#include <utility>

GFXRECON_BEGIN_NAMESPACE(gfxrecon)
GFXRECON_BEGIN_NAMESPACE(decode)

// True if the pieces, each [begin, end), cover [begin, end) without a gap.
static bool IsCovered(std::vector<std::pair<uint64_t, uint64_t>>* pieces, uint64_t begin, uint64_t end)
{
    std::sort(pieces->begin(), pieces->end());
    uint64_t covered_end = begin;
    for (const auto& piece : *pieces)
    {
        if (piece.first > covered_end)
        {
            return false;
        }
        covered_end = std::max(covered_end, piece.second);
    }
    return covered_end >= end;
}

bool PatchFillMemoryRecords(const std::vector<FillMemoryResourceValue>& values,
                            const ResourceValueSizeFunction&            get_size,
                            const ResourceValueMapFunction&             map_value,
                            std::vector<uint8_t>*                       storage,
                            std::vector<FillMemoryRecord>*              records)
{
    size_t total_size = 0;
    for (const FillMemoryRecord& record : *records)
    {
        total_size += static_cast<size_t>(record.size);
    }
    storage->resize(total_size);

    size_t copy_offset = 0;
    for (FillMemoryRecord& record : *records)
    {
        const size_t size = static_cast<size_t>(record.size);
        uint8_t*     copy = storage->data() + copy_offset;
        util::platform::MemoryCopy(copy, size, record.data, size);
        record.data = copy;
        copy_offset += size;
    }

    bool                                       all_patched = true;
    std::vector<uint8_t>                       value_bytes;
    std::vector<std::pair<uint64_t, uint64_t>> pieces;

    for (const FillMemoryResourceValue& value : values)
    {
        const uint32_t value_size = get_size(value.type);
        if ((value_size == 0) || (value.offset > std::numeric_limits<uint64_t>::max() - value_size))
        {
            all_patched = false;
            continue;
        }

        const uint64_t value_begin = value.offset;
        const uint64_t value_end   = value.offset + value_size;

        // Gather in block order, so a later record wins where records overlap, as when they are written.
        value_bytes.assign(value_size, 0);
        pieces.clear();
        for (const FillMemoryRecord& record : *records)
        {
            const uint64_t begin = std::max(value_begin, record.offset);
            const uint64_t end   = std::min(value_end, record.offset + record.size);
            if (begin < end)
            {
                const size_t size = static_cast<size_t>(end - begin);
                util::platform::MemoryCopy(
                    value_bytes.data() + (begin - value_begin), size, record.data + (begin - record.offset), size);
                pieces.emplace_back(begin, end);
            }
        }

        if (!IsCovered(&pieces, value_begin, value_end))
        {
            all_patched = false;
            continue;
        }

        map_value(value.type, value_bytes.data(), value_size);

        for (const FillMemoryRecord& record : *records)
        {
            const uint64_t begin = std::max(value_begin, record.offset);
            const uint64_t end   = std::min(value_end, record.offset + record.size);
            if (begin < end)
            {
                // record.data points into storage, so this is the same copy, but writable.
                uint8_t*     copy = storage->data() + (record.data - storage->data());
                const size_t size = static_cast<size_t>(end - begin);
                util::platform::MemoryCopy(
                    copy + (begin - record.offset), size, value_bytes.data() + (begin - value_begin), size);
            }
        }
    }

    return all_patched;
}

GFXRECON_END_NAMESPACE(decode)
GFXRECON_END_NAMESPACE(gfxrecon)
