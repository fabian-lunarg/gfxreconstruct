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

/// @file FillMemoryCommand2 meta-data block, parse -> decode round trip.

#include <catch2/catch.hpp>

#include "decode/block_buffer.h"
#include "decode/block_parser.h"
#include "format/format.h"
#include "format/format_util.h"
#include "generated/generated_vulkan_consumer.h"
#include "generated/generated_vulkan_decoder.h"
#include "util/compressor.h"

#include <cstring>
#include <memory>
#include <vector>

namespace
{

using namespace gfxrecon;

// Records every ProcessFillMemoryCommand call. It does not override ProcessFillMemoryCommand2, so the
// default per-record split is what the test sees.
class FillMemoryRecorder : public decode::VulkanConsumer
{
  public:
    struct Fill
    {
        uint64_t             memory_id;
        uint64_t             offset;
        std::vector<uint8_t> data;
    };

    void ProcessFillMemoryCommand(uint64_t memory_id, uint64_t offset, uint64_t size, const uint8_t* data) override
    {
        fills.push_back({ memory_id, offset, std::vector<uint8_t>(data, data + size) });
    }

    std::vector<Fill> fills;
};

struct TestRecord
{
    uint64_t             offset;
    std::vector<uint8_t> data;
};

struct TestModifier
{
    uint32_t             modifier_id;
    std::vector<uint8_t> data;
};

template <typename T>
void Append(std::vector<uint8_t>* bytes, const T& value)
{
    const auto* first = reinterpret_cast<const uint8_t*>(&value);
    bytes->insert(bytes->end(), first, first + sizeof(T));
}

// Data that compresses well but differs per record and per byte position.
std::vector<uint8_t> MakeData(size_t size, uint8_t seed)
{
    std::vector<uint8_t> data(size);
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<uint8_t>(seed + (i % 7));
    }
    return data;
}

std::vector<uint8_t> WriteFillMemory2Block(format::HandleId                 memory_id,
                                           const std::vector<TestModifier>& modifiers,
                                           const std::vector<TestRecord>&   records,
                                           const util::Compressor*          compressor)
{
    std::vector<uint8_t> payload;
    for (const TestModifier& modifier : modifiers)
    {
        Append(&payload, format::FillMemoryModifierHeader{ modifier.modifier_id, modifier.data.size() });
        payload.insert(payload.end(), modifier.data.begin(), modifier.data.end());
    }
    for (const TestRecord& record : records)
    {
        Append(&payload, format::FillMemoryRecordHeader{ record.offset, record.data.size() });
        payload.insert(payload.end(), record.data.begin(), record.data.end());
    }

    std::vector<uint8_t> body = payload;
    format::BlockType    type = format::BlockType::kMetaDataBlock;
    if (compressor != nullptr)
    {
        std::vector<uint8_t> compressed;
        const size_t         compressed_size = compressor->Compress(payload.size(), payload.data(), &compressed, 0);
        REQUIRE(compressed_size > 0);
        REQUIRE(compressed_size < payload.size());
        compressed.resize(compressed_size);
        body = compressed;
        type = format::BlockType::kCompressedMetaDataBlock;
    }

    format::FillMemoryCommand2Header header{};
    header.meta_header.block_header.type = type;
    header.meta_header.block_header.size = sizeof(header) - sizeof(format::BlockHeader) + body.size();
    header.meta_header.meta_data_id =
        format::MakeMetaDataId(format::ApiFamily_Vulkan, format::MetaDataType::kFillMemoryCommand2);
    header.thread_id      = 3;
    header.memory_id      = memory_id;
    header.layout_version = format::kFillMemoryCommand2LayoutVersion;
    header.modifier_count = static_cast<uint32_t>(modifiers.size());
    header.record_count   = static_cast<uint32_t>(records.size());
    header.payload_size   = payload.size();

    std::vector<uint8_t> block;
    Append(&block, header);
    block.insert(block.end(), body.begin(), body.end());
    return block;
}

std::vector<uint8_t> WriteFillMemoryBlock(format::HandleId memory_id, uint64_t offset, const std::vector<uint8_t>& data)
{
    format::FillMemoryCommandHeader header{};
    header.meta_header.block_header.type = format::BlockType::kMetaDataBlock;
    header.meta_header.block_header.size = sizeof(header) - sizeof(format::BlockHeader) + data.size();
    header.meta_header.meta_data_id =
        format::MakeMetaDataId(format::ApiFamily_Vulkan, format::MetaDataType::kFillMemoryCommand);
    header.thread_id     = 3;
    header.memory_id     = memory_id;
    header.memory_offset = offset;
    header.memory_size   = data.size();

    std::vector<uint8_t> block;
    Append(&block, header);
    block.insert(block.end(), data.begin(), data.end());
    return block;
}

// Parses one block and dispatches it to the decoder, as replay does.
void ParseAndDispatch(decode::BlockParser* parser, decode::VulkanDecoder* decoder, std::vector<uint8_t>* block)
{
    decode::BlockBuffer block_buffer;
    block_buffer.Reset(block->data(), block->size());

    decode::ParsedBlock& parsed_block = parser->ParseBlock(block_buffer);
    REQUIRE(parsed_block.IsReady());

    if (parsed_block.Holds<decode::FillMemory2Args>())
    {
        const auto& args = parsed_block.Get<decode::FillMemory2Args>();
        decoder->DispatchFillMemoryCommand2(args.command_header, args.data);
    }
    else
    {
        REQUIRE(parsed_block.Holds<decode::FillMemoryArgs>());
        const auto& args = parsed_block.Get<decode::FillMemoryArgs>();
        decoder->DispatchFillMemoryCommand(args.thread_id, args.memory_id, args.offset, args.data_size, args.data);
    }
}

void RequireFill(const FillMemoryRecorder::Fill& fill, format::HandleId memory_id, const TestRecord& record)
{
    REQUIRE(fill.memory_id == memory_id);
    REQUIRE(fill.offset == record.offset);
    REQUIRE(fill.data == record.data);
}

} // namespace

TEST_CASE("A FillMemoryCommand2 block is split into one fill per record", "[fill_memory_2]")
{
    std::unique_ptr<util::Compressor> compressor(format::CreateCompressor(format::CompressionType::kLz4));
    REQUIRE(compressor != nullptr);

    const bool compressed = GENERATE(false, true);
    INFO("compressed: " << compressed);

    // Sparse, out-of-order offsets and sizes that differ, so a stride or ordering bug shows.
    const std::vector<TestRecord> records       = { { 4096, MakeData(1000, 10) },
                                                    { 16, MakeData(3, 50) },
                                                    { 70000, MakeData(257, 90) } };
    const std::vector<uint8_t>    old_fill_data = MakeData(64, 130);

    // An old-format fill on the same memory, before and after, must still work next to the new block.
    std::vector<uint8_t> old_block_before = WriteFillMemoryBlock(12, 128, old_fill_data);
    std::vector<uint8_t> new_block = WriteFillMemory2Block(12, {}, records, compressed ? compressor.get() : nullptr);
    std::vector<uint8_t> old_block_after = WriteFillMemoryBlock(12, 256, old_fill_data);

    bool err_triggered = false;
    auto err_handler   = [&err_triggered](decode::BlockIOError, const char*) { err_triggered = true; };

    decode::BlockParser   parser(err_handler, compressor.get());
    decode::VulkanDecoder decoder;
    FillMemoryRecorder    recorder;
    decoder.AddConsumer(&recorder);

    ParseAndDispatch(&parser, &decoder, &old_block_before);
    ParseAndDispatch(&parser, &decoder, &new_block);
    ParseAndDispatch(&parser, &decoder, &old_block_after);

    REQUIRE_FALSE(err_triggered);
    REQUIRE(recorder.fills.size() == 5);
    RequireFill(recorder.fills[0], 12, { 128, old_fill_data });
    RequireFill(recorder.fills[1], 12, records[0]);
    RequireFill(recorder.fills[2], 12, records[1]);
    RequireFill(recorder.fills[3], 12, records[2]);
    RequireFill(recorder.fills[4], 12, { 256, old_fill_data });
}

TEST_CASE("A FillMemoryCommand2 reader skips modifiers by size", "[fill_memory_2]")
{
    // Modifier sizes that are not a multiple of the record header size, so a wrong skip misreads the records.
    const std::vector<TestModifier> modifiers = { { 0xffff0001, MakeData(5, 1) }, { 0xffff0002, MakeData(13, 2) } };
    const std::vector<TestRecord>   records   = { { 0, MakeData(32, 20) }, { 512, MakeData(9, 40) } };

    std::vector<uint8_t> block = WriteFillMemory2Block(7, modifiers, records, nullptr);

    bool err_triggered = false;
    auto err_handler   = [&err_triggered](decode::BlockIOError, const char*) { err_triggered = true; };

    decode::BlockParser   parser(err_handler, nullptr);
    decode::VulkanDecoder decoder;
    FillMemoryRecorder    recorder;
    decoder.AddConsumer(&recorder);

    ParseAndDispatch(&parser, &decoder, &block);

    REQUIRE_FALSE(err_triggered);
    REQUIRE(recorder.fills.size() == 2);
    RequireFill(recorder.fills[0], 7, records[0]);
    RequireFill(recorder.fills[1], 7, records[1]);
}

TEST_CASE("A truncated FillMemoryCommand2 block is dropped whole", "[fill_memory_2]")
{
    const std::vector<TestRecord> records = { { 0, MakeData(32, 20) }, { 512, MakeData(9, 40) } };
    std::vector<uint8_t>          block   = WriteFillMemory2Block(7, {}, records, nullptr);

    // Claim one more record than the payload holds. The first two are intact, but none may be written.
    format::FillMemoryCommand2Header header{};
    std::memcpy(&header, block.data(), sizeof(header));
    ++header.record_count;
    std::memcpy(block.data(), &header, sizeof(header));

    decode::BlockParser   parser([](decode::BlockIOError, const char*) {}, nullptr);
    decode::VulkanDecoder decoder;
    FillMemoryRecorder    recorder;
    decoder.AddConsumer(&recorder);

    ParseAndDispatch(&parser, &decoder, &block);

    REQUIRE(recorder.fills.empty());
}
