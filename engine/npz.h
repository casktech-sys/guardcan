// npz.h — minimal, dependency-free reader for NumPy .npz archives written by np.savez.
//
// An .npz is a zip of .npy files. np.savez stores them uncompressed (method 0), so reading one is:
// find the zip central directory -> locate each member's data -> parse the .npy header (dtype,
// shape, C order) -> take the raw little-endian bytes. Zip64 records (numpy forces them on members)
// are handled. Compressed archives (np.savez_compressed) are rejected with a clear error.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace npz {

struct Array {
    std::string dtype;            // numpy descr, e.g. "<f4", "<i4", "|i1"
    std::vector<size_t> shape;
    std::vector<char> data;       // raw bytes, C order

    size_t size() const {
        size_t n = 1;
        for (size_t d : shape) n *= d;
        return n;
    }

    // Convert to float32 (supports <f4, <f8, <i4, <i8, |i1, |u1).
    std::vector<float> to_float() const {
        std::vector<float> out(size());
        const char* p = data.data();
        for (size_t i = 0; i < out.size(); ++i) {
            if (dtype == "<f4")      { float v;   std::memcpy(&v, p + 4 * i, 4); out[i] = v; }
            else if (dtype == "<f8") { double v;  std::memcpy(&v, p + 8 * i, 8); out[i] = (float)v; }
            else if (dtype == "<i4") { int32_t v; std::memcpy(&v, p + 4 * i, 4); out[i] = (float)v; }
            else if (dtype == "<i8") { int64_t v; std::memcpy(&v, p + 8 * i, 8); out[i] = (float)v; }
            else if (dtype == "|i1") { out[i] = (float)(int8_t)p[i]; }
            else if (dtype == "|u1") { out[i] = (float)(uint8_t)p[i]; }
            else throw std::runtime_error("npz: unsupported dtype " + dtype);
        }
        return out;
    }
};

using Archive = std::map<std::string, Array>;

namespace detail {

inline uint16_t rd16(const char* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint32_t rd32(const char* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t rd64(const char* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

inline size_t itemsize(const std::string& descr) {
    size_t i = descr.size();
    while (i > 0 && descr[i - 1] >= '0' && descr[i - 1] <= '9') --i;
    if (i == descr.size()) throw std::runtime_error("npz: bad descr " + descr);
    return std::stoul(descr.substr(i));
}

// value following 'key': in the .npy header dict (a Python literal)
inline std::string header_field(const std::string& h, const std::string& key) {
    size_t k = h.find("'" + key + "'");
    if (k == std::string::npos) throw std::runtime_error("npz: header missing " + key);
    size_t c = h.find(':', k);
    size_t s = h.find_first_not_of(' ', c + 1);
    if (h[s] == '\'') return h.substr(s + 1, h.find('\'', s + 1) - s - 1);
    if (h[s] == '(') return h.substr(s + 1, h.find(')', s) - s - 1);
    size_t e = h.find_first_of(",}", s);
    return h.substr(s, e - s);
}

inline Array parse_npy(const char* p, size_t n, const std::string& name) {
    if (n < 10 || std::memcmp(p, "\x93NUMPY", 6) != 0)
        throw std::runtime_error("npz: member " + name + " is not a .npy file");
    const uint8_t major = (uint8_t)p[6];
    const size_t hlen = (major == 1) ? rd16(p + 8) : rd32(p + 8);
    const size_t hoff = (major == 1) ? 10 : 12;
    const std::string h(p + hoff, hlen);

    Array a;
    a.dtype = header_field(h, "descr");
    if (header_field(h, "fortran_order").find("False") == std::string::npos)
        throw std::runtime_error("npz: " + name + " is Fortran-ordered; expected C order");
    const std::string shp = header_field(h, "shape");
    size_t pos = 0;
    while (pos < shp.size()) {
        size_t e = shp.find(',', pos);
        std::string tok = shp.substr(pos, e == std::string::npos ? std::string::npos : e - pos);
        if (tok.find_first_of("0123456789") != std::string::npos) a.shape.push_back(std::stoul(tok));
        if (e == std::string::npos) break;
        pos = e + 1;
    }
    const size_t bytes = a.size() * itemsize(a.dtype);
    if (hoff + hlen + bytes > n) throw std::runtime_error("npz: truncated member " + name);
    a.data.assign(p + hoff + hlen, p + hoff + hlen + bytes);
    return a;
}

}  // namespace detail

inline Archive load(const std::string& path) {
    using namespace detail;
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npz: cannot open " + path);
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const char* b = buf.data();
    const size_t n = buf.size();

    // End-of-central-directory record (scan back over a possible trailing comment).
    if (n < 22) throw std::runtime_error("npz: file too small " + path);
    size_t eocd = n - 22;
    while (eocd > 0 && rd32(b + eocd) != 0x06054b50u) --eocd;
    if (rd32(b + eocd) != 0x06054b50u) throw std::runtime_error("npz: not a zip file " + path);
    uint64_t entries = rd16(b + eocd + 10);
    uint64_t cd_off = rd32(b + eocd + 16);
    if (cd_off == 0xFFFFFFFFu || entries == 0xFFFFu) {           // zip64 end record
        const size_t loc = eocd - 20;
        if (rd32(b + loc) != 0x07064b50u) throw std::runtime_error("npz: bad zip64 locator");
        const size_t z64 = rd64(b + loc + 8);
        entries = rd64(b + z64 + 32);
        cd_off = rd64(b + z64 + 48);
    }

    Archive out;
    size_t p = cd_off;
    for (uint64_t e = 0; e < entries; ++e) {
        if (rd32(b + p) != 0x02014b50u) throw std::runtime_error("npz: bad central directory");
        const uint16_t method = rd16(b + p + 10);
        uint64_t usize = rd32(b + p + 24);
        const uint16_t nlen = rd16(b + p + 28), elen = rd16(b + p + 30), clen = rd16(b + p + 32);
        uint64_t lho = rd32(b + p + 42);
        std::string name(b + p + 46, nlen);
        // zip64 extra field: 64-bit values replace whichever 32-bit fields are 0xFFFFFFFF
        for (size_t x = p + 46 + nlen; x + 4 <= p + 46 + nlen + elen;) {
            const uint16_t id = rd16(b + x), sz = rd16(b + x + 2);
            if (id == 0x0001) {
                size_t q = x + 4;
                if (usize == 0xFFFFFFFFu) { usize = rd64(b + q); q += 8; }
                if (rd32(b + p + 20) == 0xFFFFFFFFu) q += 8;       // compressed size
                if (lho == 0xFFFFFFFFu) { lho = rd64(b + q); }
            }
            x += 4 + sz;
        }
        if (method != 0)
            throw std::runtime_error("npz: " + name + " is compressed; write it with np.savez, not savez_compressed");
        if (rd32(b + lho) != 0x04034b50u) throw std::runtime_error("npz: bad local header for " + name);
        const size_t data = lho + 30 + rd16(b + lho + 26) + rd16(b + lho + 28);
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0) name.resize(name.size() - 4);
        out[name] = parse_npy(b + data, (size_t)usize, name);
        p += 46 + nlen + elen + clen;
    }
    return out;
}

inline const Array& get(const Archive& a, const std::string& key) {
    auto it = a.find(key);
    if (it == a.end()) throw std::runtime_error("npz: missing array '" + key + "'");
    return it->second;
}

}  // namespace npz
