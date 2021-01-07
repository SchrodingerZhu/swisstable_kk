#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <xmmintrin.h>

#if defined(__GNUC__) || defined(__clang__)
#  define PURE __attribute__((const))
#  define FAST_PATH inline __attribute__((always_inline))
#  define likely(x) __builtin_expect(!!(x), 1)
#  define unlikely(x) __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
#  define PURE 
#  define FAST_PATH inline __forceinline
#  define likely(x) !!(x)
#  define unlikely(x) !!(x)
#else
#  define PURE
#  define FAST_PATH inline
#  define likely(x) !!(x)
#  define unlikely(x) !!(x)
#endif 

#define EMPTY   0b11111111u
#define DELETED 0b10000000u

#ifdef __SSE2__

#define BITMASK_STRIDE 0x1u
#define BITMASK_MASK   0xffffu

typedef uint16_t bitmask_t;
typedef __m128i  group_t; 

static FAST_PATH group_t load(const void * __restrict__ ptr) {
  return _mm_loadu_si128((const group_t*)ptr);
}

static FAST_PATH group_t load_aligned(const void * __restrict__ ptr) {
  return _mm_load_si128((const group_t*)ptr);
}

static FAST_PATH void store_aligned(group_t group, void * __restrict__ ptr) {
  return _mm_store_si128((group_t*)ptr, group);
}

static FAST_PATH PURE bitmask_t match_byte(group_t group, uint8_t byte) {
  group_t cmp = _mm_cmpeq_epi8(group, _mm_set1_epi8(byte));
  return _mm_movemask_epi8(cmp);
}

static FAST_PATH PURE bitmask_t match_empty(group_t group) {
  return match_byte(group, EMPTY);
}

static FAST_PATH PURE bitmask_t match_empty_or_deleted(group_t group) {
  return _mm_movemask_epi8(group);
}

static FAST_PATH PURE bitmask_t match_full(group_t group) {
  return match_empty_or_deleted(group) ^ BITMASK_MASK;
}

/// Performs the following transformation on all bytes in the group:
/// - `EMPTY => EMPTY`
/// - `DELETED => EMPTY`
/// - `FULL => DELETED`

static FAST_PATH PURE group_t convert_special_to_empty_and_full_to_deleted(group_t group) {
  group_t zero = _mm_setzero_si128();
  group_t special =_mm_cmpgt_epi8(zero, group);
  return _mm_or_si128(special, _mm_set1_epi8(0x80u));
}

#else 

#define BITMASK_STRIDE 0x8ull
#define BITMASK_MASK   0x8080808080808080ull

#  if defined(__POINTER_WIDTH__)
#    if __POINTER_WIDTH__ == 64
typedef uint64_t group_t;
#    else 
typedef uint32_t group_t;
#    endif 
#  elif defined(_WIN64) || defined(__x86_64__) || defined(__amd64) || defined(__amd64__) || defined(__aarch64__) || defined(_M_ARM64) || defined(__arm64__)
#    define __POINTER_WIDTH__ 64
typedef uint64_t group_t;
#  else
#    define __POINTER_WIDTH__ 32
typedef uint32_t group_t;
#  endif

typedef group_t bitmask_t;

static FAST_PATH PURE group_t repeat(group_t byte) {
  byte = (byte <<  8u) | byte;
  byte = (byte << 16u) | byte;
#  if __POINTER_WIDTH__ > 32
  byte = (byte << 32u) | byte;
#  endif
  return byte;
}

static FAST_PATH PURE group_t bswap(group_t group) {
#  if __POINTER_WIDTH__ > 32
#    ifdef _MSC_VER
  return _byteswap_uint64(group);
#    elif defined(__GNUC__) || defined(__clang__)
  return __builtin_bswap64(group);
#    else
  group = ((0x00ff00ff00ff00ffull & group) <<  8) | (((~0x00ff00ff00ff00ffull) & group) >>  8);
  group = ((0x0000ffff0000ffffull & group) << 16) | (((~0x0000ffff0000ffffull) & group) >> 16);
  group = ((0x00000000ffffffffull & group) << 32) | (((~0x00000000ffffffffull) & group) >> 32);
  return group;
#    endif
#  else
#    ifdef _MSC_VER
  return _byteswap_ulong(group);
#    elif defined(__GNUC__) || defined(__clang__)
  return __builtin_bswap32(group);
#    else
  group = ((0x00ff00ffull & group) <<  8) | (((~0x00ff00ffull) & group) >>  8);
  group = ((0x0000ffffull & group) << 16) | (((~0x0000ffffull) & group) >> 16);
  return group;
#    endif
#  endif
}

static FAST_PATH PURE group_t little_endian(group_t group) {
  if (*(uint16_t *)"\0\xff" < 0x100) {
    return bswap(group);
  } else {
    return group;
  }
}

static FAST_PATH group_t load(const void * __restrict__ ptr) {
  group_t group;
  memcpy(&group, ptr, sizeof(group));
  return group;
}

static FAST_PATH group_t load_aligned(const void * __restrict__ ptr) {
  return *(const group_t*)ptr;
}

static FAST_PATH void store_aligned(group_t group, void * __restrict__ ptr) {
  *(group_t*)ptr = group;
}

static FAST_PATH PURE bitmask_t match_byte(group_t group, uint8_t byte) {
  group_t cmp = group ^ repeat(byte);
  return little_endian((cmp - repeat(0x01)) & ~cmp & repeat(0x80));
}

static FAST_PATH PURE bitmask_t match_empty(group_t group) {
  return little_endian(group & (group << 1) & repeat(0x80));
}

static FAST_PATH PURE bitmask_t match_empty_or_deleted(group_t group) {
  return little_endian(group & repeat(0x80));
}

static FAST_PATH PURE bitmask_t match_full(group_t group) {
  return match_empty_or_deleted(group) ^ BITMASK_MASK;
}

/// Performs the following transformation on all bytes in the group:
/// - `EMPTY => EMPTY`
/// - `DELETED => EMPTY`
/// - `FULL => DELETED`

static FAST_PATH PURE group_t convert_special_to_empty_and_full_to_deleted(group_t group) {
  group_t full = ~group & repeat(0x80);
  return ~full + (full >> 7);
}

#endif

static FAST_PATH PURE bitmask_t invert(bitmask_t mask) {
  return mask ^ BITMASK_MASK;
}

static FAST_PATH PURE bitmask_t remove_lowest_bit(bitmask_t mask) {
  return mask & (mask - 1);
}

static FAST_PATH PURE bitmask_t any_bit_set(bitmask_t mask) {
  return mask != 0;
}

static FAST_PATH PURE size_t trailing_zeros(bitmask_t mask) {
# ifdef _MSC_VER
  unsigned long idx;
  _BitScanForward(&idx, mask);
  return idx / BITMASK_STRIDE;
# else
  return __builtin_ctz(mask) / BITMASK_STRIDE;
# endif
}

static FAST_PATH PURE size_t lowest_set_bit(bitmask_t mask) {
  if (mask == 0) { return (size_t)-1; }
  else {
    return trailing_zeros(mask);
  }
}

static FAST_PATH PURE size_t lowest_set_bit_nonzero(bitmask_t mask) {
    return trailing_zeros(mask);
}

static FAST_PATH PURE size_t leading_zeros(bitmask_t mask) {
# ifdef _MSC_VER
  unsigned long idx;
  _BitScanReverse(&idx, mask);
  return idx / BITMASK_STRIDE;
# else
  return __builtin_clz(mask) / BITMASK_STRIDE;
# endif
}

typedef bitmask_t mask_iter_t;

static FAST_PATH size_t next(mask_iter_t* iter) {
    size_t bit = lowest_set_bit(*iter);
    *iter = remove_lowest_bit(*iter);
    return bit;
}

static FAST_PATH PURE bool is_full(uint8_t ctrl) {
    return (ctrl & 0x80) == 0;
}

static FAST_PATH PURE bool is_special(uint8_t ctrl) {
    return (ctrl & 0x80) != 0;
}

static FAST_PATH PURE bool special_is_empty(uint8_t ctrl) {
    return (ctrl & 0x01) != 0;
}

static FAST_PATH PURE size_t h1(uint64_t hash) {
    return hash;
}

static FAST_PATH PURE uint8_t h2(uint64_t hash) {
    const static size_t HASH_LEN = 
        sizeof(size_t) < sizeof(uint64_t) ? sizeof(size_t) : sizeof(uint64_t);
    return (hash >> (HASH_LEN * 8 - 7)) & 0x7f;
}

typedef struct probe_seq_s {
  size_t pos;
  size_t stride;
} probe_seq_t;

static FAST_PATH void move_next(probe_seq_t * seq, size_t bucket_mask) {
  seq->stride += sizeof(group_t);
  seq->pos    += seq->stride;
  seq->pos    &= bucket_mask;
}

static FAST_PATH PURE size_t next_power_of_two(size_t val) {
  size_t idx;
# ifdef _MSC_VER
  _BitScanReverse(&idx, val - 1);
# else
  idx = __builtin_clz(val - 1);
# endif
  return 1ull << ((8ull * sizeof(size_t)) -  idx);
}

static FAST_PATH PURE size_t capacity_to_buckets(size_t cap) {
  if (cap < 8) {
    return (cap < 4) ? 4 : 8;
  }
  return next_power_of_two(cap * 8);
}

static FAST_PATH PURE size_t bucket_mask_to_capacity(size_t bucket_mask) {
  if (bucket_mask < 8) {
    return bucket_mask;
  } else {
    return (bucket_mask + 1) / 8 * 7;
  }
}

static FAST_PATH PURE uint8_t* static_empty_grp() {
  static _Alignas(group_t) uint8_t 
    _EMPTY[sizeof(group_t)] = {[0 ... sizeof(group_t) - 1] = EMPTY};
  return _EMPTY;
}

static FAST_PATH size_t calculate_layout(size_t buckets, size_t* offset, size_t* alignment) {
  *alignment = 
    _Alignof(kk_box_t) > _Alignof(group_t) ? _Alignof(kk_box_t) : _Alignof(group_t);
  *offset = (buckets * sizeof(kk_box_t) + *alignment - 1) & ~(*alignment - 1);
  return *offset + buckets + sizeof(group_t);
}

typedef struct table_s {
  size_t bucket_mask;
  uint8_t *ctrl;
  size_t growth_left;
  size_t items;
} table_t;

static FAST_PATH size_t num_ctrl_bytes(table_t* table) {
  return table->bucket_mask + 1 + sizeof(group_t);
}

static FAST_PATH PURE table_t
new_table() {
  table_t table;
  table.ctrl        = static_empty_grp();
  table.bucket_mask = 0;
  table.items       = 0;
  table.growth_left = 0;
  return table;
}

static FAST_PATH table_t
new_table_uninitialized(size_t buckets, kk_context_t* ctx) {
  table_t result;
  size_t offset, alignment;
  size_t size = calculate_layout(buckets, &offset, &alignment);
  uint8_t * ptr = (uint8_t *)mi_heap_malloc_aligned(ctx->heap, size, alignment);
  uint8_t * ctrl = ptr + offset;
  result.bucket_mask = buckets - 1;
  result.ctrl = ctrl;
  result.growth_left = bucket_mask_to_capacity(buckets - 1);
  result.items = 0;
  return result;
}

static FAST_PATH table_t
new_table_with_capacity(size_t capacity, kk_context_t* ctx) {
  if (capacity == 0) {
    return new_table();
  } else {
    size_t buckets = capacity_to_buckets(capacity);
    table_t result = new_table_uninitialized(buckets, ctx);
    memset(result.ctrl, EMPTY, num_ctrl_bytes(&result));
    return result;
  }
}

static FAST_PATH PURE kk_box_t* bucket_from_baseidx(kk_box_t* base, size_t index) {
  return base - index;
}

static FAST_PATH PURE size_t bucket_to_baseidx(kk_box_t* bucket, kk_box_t* base) {
  return bucket - base;
}

static FAST_PATH PURE size_t buckets(table_t* table) {
  return table->bucket_mask + 1;
}

static FAST_PATH PURE kk_box_t* bucket_nextn(kk_box_t* bucket, size_t offset) {
  return bucket - offset;
}

static FAST_PATH void free_buckets(table_t* table) {
  // inner boxes are not dropped
  size_t offset, alignment;
  size_t size = calculate_layout(buckets(table), &offset, &alignment);
  KK_UNUSED(alignment);
  kk_free(table->ctrl - offset);
}

static FAST_PATH kk_box_t* data_end(table_t* table) {
  return (kk_box_t*)table->ctrl;
}

static FAST_PATH kk_box_t* data_start(table_t* table) {
  return data_end(table) - buckets(table);
}

static FAST_PATH kk_box_t* bucket(table_t* table, size_t index) {
  return bucket_from_baseidx(data_end(table), index);
}

static FAST_PATH size_t bucket_index(table_t* table, kk_box_t* bucket) {
  return bucket_to_baseidx(bucket, data_end(table));
}

static FAST_PATH void set_ctrl(table_t* table, size_t index, uint8_t ctrl) {
  size_t index2 = ((index - sizeof(group_t)) & table->bucket_mask) + sizeof(group_t);
  table->ctrl[index] = ctrl;
  table->ctrl[index2] = ctrl;
}

static FAST_PATH void erase(table_t* table, kk_box_t* item, kk_context_t* ctx) {
  size_t index = bucket_index(table, item);
  size_t index_before = (index - sizeof(group_t)) & table->bucket_mask;
  bitmask_t empty_before = match_empty(load(table->ctrl + index_before));
  bitmask_t empty_after = match_empty(load(table->ctrl + index));
  uint8_t ctrl;
  if (leading_zeros(empty_before) + trailing_zeros(empty_before) >= sizeof(group_t)) {
    ctrl = DELETED;
  } else {
    table->growth_left++;
    ctrl = EMPTY;
  };
  set_ctrl(table, index, ctrl);
  table->items--;
  kk_box_drop(*item, ctx);
}

static FAST_PATH probe_seq_t probe_seq(table_t* table, uint64_t hash) {
  probe_seq_t seq;
  seq.pos = h1(hash) & table->bucket_mask;
  seq.stride = 0;
  return seq;
}

static FAST_PATH size_t find_insert_slot(table_t* table, uint64_t hash) {
  probe_seq_t seq = probe_seq(table, hash);
  while (true) {
    group_t group = load(table->ctrl + seq.pos);
    size_t bit =  lowest_set_bit(match_empty_or_deleted(group));
    if (bit != (size_t)-1) {
      size_t result = (seq.pos + bit) & table->bucket_mask;
      if (unlikely(is_full(table->ctrl[result]))) {
        return lowest_set_bit_nonzero(match_empty_or_deleted(load_aligned(table->ctrl)));
      } else {
        return result;
      }
    }
    move_next(&seq, table->bucket_mask);
  }
}

static FAST_PATH bool is_empty_singleton(table_t* table) {
  return table->bucket_mask == 0;
}

static FAST_PATH void clear_no_drop(table_t* table) {
  if (!is_empty_singleton(table)) {
    memset(table->ctrl, EMPTY, num_ctrl_bytes(table));
  }
  table->items = 0;
  table->growth_left = bucket_mask_to_capacity(table->bucket_mask);
}

typedef struct table_iter_s {
  bitmask_t current_group;
  kk_box_t* data;
  const uint8_t* next_ctrl;
  const uint8_t* end;
} table_iter_t;

static FAST_PATH table_iter_t 
new_table_iter(const uint8_t* ctrl, kk_box_t* data, size_t length) {
  table_iter_t result;
  const uint8_t* end = ctrl + length;
  bitmask_t current_group = match_full(load_aligned(ctrl));
  const uint8_t* next_ctrl = ctrl + sizeof(group_t);
  result.current_group = current_group;
  result.data = data;
  result.next_ctrl = next_ctrl;
  result.end = end;
  return result;
}

static FAST_PATH kk_box_t* next_table_item(table_iter_t* iter) {
  while ( true ) {
    size_t index = lowest_set_bit(iter->current_group);
    if (index != (size_t)-1) {
      iter->current_group = remove_lowest_bit(iter->current_group);
      return bucket_nextn(iter->data, index);
    }

    if (iter->next_ctrl >= iter->end) {
      return NULL;
    }

    iter->current_group = match_full(load_aligned(iter->next_ctrl));
    iter->data = bucket_nextn(iter->data, sizeof(group_t));
    iter->next_ctrl += sizeof(group_t);
  }
}

static FAST_PATH table_iter_t table_iterator(table_t* table) {
  kk_box_t* data = bucket_from_baseidx(data_end(table), 0);
  return new_table_iter(table->ctrl, data, buckets(table));
}

static FAST_PATH void clear(table_t* table, kk_context_t* ctx) {
  table_iter_t iter = table_iterator(table);
  kk_box_t* bucket = next_table_item(&iter);
  while (bucket) {
    kk_box_drop(*bucket, ctx);
    bucket = next_table_item(&iter);
  }
  clear_no_drop(table);
}
