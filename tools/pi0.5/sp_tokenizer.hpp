#pragma once

#include <sentencepiece_processor.h>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>

// Simple SentencePiece tokenizer wrapper for PaliGemma
// Uses tokenizer.model file directly (same as OpenPI)
class SpTokenizer {
public:
    SpTokenizer() = default;
    ~SpTokenizer() = default;

    // Initialize from tokenizer.model file path
    bool init(const std::string& model_path) {
        auto status = processor_.Load(model_path);
        if (!status.ok()) {
            fprintf(stderr, "Failed to load SentencePiece model from %s: %s\n",
                    model_path.c_str(), status.ToString().c_str());
            return false;
        }

        // Get special token IDs
        bos_id_ = processor_.bos_id();
        eos_id_ = processor_.eos_id();
        pad_id_ = processor_.pad_id();
        unk_id_ = processor_.unk_id();

        printf("Loaded SentencePiece model: vocab_size=%d, bos=%d, eos=%d, pad=%d\n",
               processor_.GetPieceSize(), bos_id_, eos_id_, pad_id_);

        initialized_ = true;
        return true;
    }

    // Encode text to token IDs
    std::vector<int32_t> encode(const std::string& text, bool add_bos = true) {
        if (!initialized_) {
            throw std::runtime_error("SpTokenizer not initialized");
        }

        std::vector<int> ids;
        auto status = processor_.Encode(text, &ids);
        if (!status.ok()) {
            throw std::runtime_error("SentencePiece encode failed: " + status.ToString());
        }

        std::vector<int32_t> result;
        if (add_bos && bos_id_ >= 0) {
            result.push_back(bos_id_);
        }
        for (int id : ids) {
            result.push_back(static_cast<int32_t>(id));
        }
        return result;
    }

    // Decode token IDs to text
    std::string decode(const std::vector<int32_t>& ids) {
        if (!initialized_) {
            throw std::runtime_error("SpTokenizer not initialized");
        }

        std::vector<int> int_ids(ids.begin(), ids.end());
        std::string text;
        auto status = processor_.Decode(int_ids, &text);
        if (!status.ok()) {
            throw std::runtime_error("SentencePiece decode failed: " + status.ToString());
        }
        return text;
    }

    // Get vocabulary size
    int vocab_size() const {
        return initialized_ ? processor_.GetPieceSize() : 0;
    }

    // Convert token ID to string piece
    std::string id_to_piece(int32_t id) {
        if (!initialized_) return "";
        return processor_.IdToPiece(id);
    }

    // Convert string piece to token ID
    int32_t piece_to_id(const std::string& piece) {
        if (!initialized_) return -1;
        return processor_.PieceToId(piece);
    }

    // Special token IDs
    int32_t bos_id() const { return bos_id_; }
    int32_t eos_id() const { return eos_id_; }
    int32_t pad_id() const { return pad_id_; }
    int32_t unk_id() const { return unk_id_; }

    bool is_initialized() const { return initialized_; }

private:
    sentencepiece::SentencePieceProcessor processor_;
    bool initialized_ = false;
    int32_t bos_id_ = -1;
    int32_t eos_id_ = -1;
    int32_t pad_id_ = -1;
    int32_t unk_id_ = -1;
};
