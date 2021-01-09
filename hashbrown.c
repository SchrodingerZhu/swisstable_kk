#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>

#if defined(__GNUC__) || defined(__clang__)
#  define PURE //__attribute__((const))
#  define FAST_PATH //inline __attribute__((always_inline, hot))
#  define COLD_PATH //__attribute__((noinline, cold))
#  define likely(x) __builtin_expect(!!(x), 1)
#  define unlikely(x) __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
#  define PURE 
#  define FAST_PATH inline __forceinline
#  define COLD_PATH __declspec(noinline)
#  define likely(x) !!(x)
#  define unlikely(x) !!(x)
#else
#  define PURE
#  define FAST_PATH inline
#  define COLD_PATH 
#  define likely(x) !!(x)
#  define unlikely(x) !!(x)
#endif 

#define EMPTY   0b11111111u
#define DELETED 0b10000000u

#ifdef __SSE2__

#include <xmmintrin.h>
#define BITMASK_STRIDE 0x1u
#define BITMASK_MASK   0xffffu

typedef uint16_t bitmask_t;
typedef __m128i  group_t; 

static FAST_PATH group_t load(const void * ptr) {
  return _mm_loadu_si128((const group_t*)ptr);
}

static FAST_PATH group_t load_aligned(const void * ptr) {
  return _mm_load_si128((const group_t*)ptr);
}

static FAST_PATH void store_aligned(group_t group, void * ptr) {
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

static FAST_PATH group_t load(const void * ptr) {
  group_t group;
  memcpy(&group, ptr, sizeof(group));
  return group;
}

static FAST_PATH group_t load_aligned(const void * ptr) {
  return *(const group_t*)ptr;
}

static FAST_PATH void store_aligned(group_t group, void * ptr) {
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

typedef bitmask_t bitmask_iter_t;

static FAST_PATH size_t next_mask(bitmask_iter_t* iter) {
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
    static const size_t HASH_LEN = 
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
  kk_function_t hasher;
  kk_function_t comparator;
} table_t;

static FAST_PATH size_t num_ctrl_bytes(table_t* table) {
  return table->bucket_mask + 1 + sizeof(group_t);
}

static FAST_PATH PURE table_t
new_table(kk_function_t hasher, kk_function_t comparator) {
  table_t table;
  table.ctrl        = static_empty_grp();
  table.bucket_mask = 0;
  table.items       = 0;
  table.growth_left = 0;
  table.hasher      = hasher;
  table.comparator  = comparator;
  return table;
}

static FAST_PATH table_t
new_table_uninitialized(size_t buckets, kk_function_t hasher, kk_function_t comparator, kk_context_t* ctx) {
  table_t result;
  size_t offset, alignment;
  size_t size = calculate_layout(buckets, &offset, &alignment);
  uint8_t * ptr = (uint8_t *)mi_heap_malloc_aligned(ctx->heap, size, alignment);
  uint8_t * ctrl = ptr + offset;
  result.bucket_mask = buckets - 1;
  result.ctrl = ctrl;
  result.growth_left = bucket_mask_to_capacity(buckets - 1);
  result.items = 0;
  result.hasher = hasher;
  result.comparator = comparator;
  return result;
}

static FAST_PATH table_t
new_table_with_capacity(size_t capacity, kk_function_t hasher, kk_function_t comparator, kk_context_t* ctx) {
  if (capacity == 0) {
    return new_table(hasher, comparator);
  } else {
    size_t buckets = capacity_to_buckets(capacity);
    table_t result = new_table_uninitialized(buckets, hasher, comparator, ctx);
    memset(result.ctrl, EMPTY, num_ctrl_bytes(&result));
    return result;
  }
}

static FAST_PATH PURE kk_box_t* bucket_from_baseidx(kk_box_t* base, size_t index) {
  return base - index;
}

static FAST_PATH PURE size_t bucket_to_baseidx(kk_box_t* bucket, kk_box_t* base) {
  return base - bucket;
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

typedef kk_box_t* bucket_t;

static FAST_PATH kk_box_t read_bucket(bucket_t b) {
  return *(b - 1);
}

static FAST_PATH void write_bucket(bucket_t b, kk_box_t value) {
  *(b - 1) = value;
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

static FAST_PATH void erase(table_t* table, bucket_t item, kk_context_t* ctx) {
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
  kk_box_drop(read_bucket(item), ctx);
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
  bucket_t bucket = next_table_item(&iter);
  while (bucket) {
    kk_box_drop(read_bucket(bucket), ctx);
    bucket = next_table_item(&iter);
  }
  clear_no_drop(table);
}

// FIXME: currently Koka do not have a good uint64 support
//        use size_t instead for now; this will be problematic
//        for 32bit machines
#define apply_hash(h, data, ctx) \
  kk_function_call(size_t, \
  (kk_function_t, kk_box_t, kk_context_t*), \
  kk_function_dup(h), \
  (h, data, ctx))

#define apply_cmp(h, a, b, ctx) \
  kk_function_call(bool, \
  (kk_function_t, kk_box_t, kk_box_t, kk_context_t*), \
  kk_function_dup(h), \
  (h, a, b, ctx))

static FAST_PATH void 
resize(table_t* table, size_t capacity, kk_context_t* ctx) {
  table_t new_table = new_table_with_capacity(capacity, table->hasher, table->comparator, ctx);
  new_table.growth_left -= table->items;
  new_table.items = table->items;

  table_iter_t iter = table_iterator(table);
  bucket_t item = next_table_item(&iter);
  while (item) {
    uint64_t hash_value = apply_hash(table->hasher, read_bucket(item), ctx);
    size_t index = find_insert_slot(&new_table, hash_value);
    set_ctrl(&new_table, index, h2(hash_value));
    write_bucket(bucket(&new_table, index), read_bucket(item));
    item = next_table_item(&iter);
  }
  if (!is_empty_singleton(table)) {
    free_buckets(table);
  }
  *table = new_table;
}

static FAST_PATH void 
shrink_to(table_t* table, size_t min_size, kk_context_t* ctx) {
  min_size = min_size < table->items ? table->items : min_size;
  if (min_size == 0) {
    *table = new_table(table->hasher, table->comparator);
    return;
  }

  size_t min_buckets = capacity_to_buckets(min_size);

  if (min_buckets < buckets(table)) {
    if (table->items == 0) {
      *table = new_table_with_capacity(min_size, table->hasher, table->comparator, ctx);
    } else {
      resize(table, min_size, ctx);
    }
  }
}

#define probe_index(idx) \
  (((idx - probe_seq(table, hash).pos) & table->bucket_mask) / sizeof(group_t))

static FAST_PATH void rehash_in_place(table_t* table, kk_context_t* ctx) {
  for (size_t i = 0, end = buckets(table); i < end; i += sizeof(group_t)) {
    group_t group = 
      convert_special_to_empty_and_full_to_deleted(load_aligned(table->ctrl + i));
    store_aligned(group, table->ctrl + i);
  }
  size_t bkts = buckets(table);
  if (bkts < sizeof(group_t)) {
    memmove(table->ctrl + sizeof(group_t), table->ctrl, bkts);
  } else {
    memmove(table->ctrl + bkts, table->ctrl, sizeof(group_t));
  }
  
  for (size_t i = 0; i < bkts; ++i) {
    if (table->ctrl[i] != DELETED) {
      continue;
    }
    while (true) {
      bucket_t item = bucket(table, i);
      uint64_t hash = apply_hash(table->hasher, read_bucket(item), ctx);
      size_t new_idx = find_insert_slot(table, hash);
      uint8_t prev_ctrl = table->ctrl[new_idx];

      if (likely(probe_index(i) == probe_index(new_idx))) {
        set_ctrl(table, i, h2(hash));
        goto rehash_outer_loop;
      }

      set_ctrl(table, new_idx, h2(hash));
      
      if (prev_ctrl == EMPTY) {
        set_ctrl(table, i, EMPTY);
        write_bucket(bucket(table, new_idx), read_bucket(item));
        goto rehash_outer_loop;
      } else {
        kk_box_t tmp;
        tmp = read_bucket(item);
        write_bucket(item, read_bucket(bucket(table, new_idx)));
        write_bucket(bucket(table, new_idx), tmp);
      }
    }
    rehash_outer_loop: continue;
  }

  table->growth_left = bucket_mask_to_capacity(table->bucket_mask) - table->items;
}

static COLD_PATH 
void reserve_rehash(table_t* table, size_t additional, kk_context_t* ctx) {
  size_t new_items = table->items + additional;
  size_t full_capacity = bucket_mask_to_capacity(table->bucket_mask);
  if (new_items <= full_capacity / 2) {
    rehash_in_place(table, ctx);
  } else {
    resize(
      table, 
      new_items > full_capacity + 1 ? new_items : full_capacity + 1,
      ctx
    );
  }
}

static FAST_PATH
void reserve(table_t* table, size_t additional, kk_context_t* ctx) {
  if (additional > table->growth_left) {
    reserve_rehash(table, additional, ctx);
  }
}

static FAST_PATH bucket_t
insert(table_t* table, uint64_t hash, kk_box_t value, kk_context_t* ctx) {
  size_t idx = find_insert_slot(table, hash);
  uint8_t old_ctrl = table->ctrl[idx];
  if (unlikely(table->growth_left == 0 && special_is_empty(old_ctrl))) {
    reserve(table, 1, ctx);
    idx = find_insert_slot(table, hash);
  }
  bucket_t bkt = bucket(table, idx);
  table->growth_left -= special_is_empty(old_ctrl);
  set_ctrl(table, idx, h2(hash));
  write_bucket(bkt, value);
  table->items += 1;
  return bkt;
}

static FAST_PATH void
replace_bucket(bucket_t bucket, kk_box_t value, kk_context_t* ctx) {
  kk_box_drop(read_bucket(bucket), ctx);
  write_bucket(bucket, value);
}

typedef struct table_iterhash_s {
  table_t * table;
  uint8_t h2_hash;
  probe_seq_t probe_seq;
  group_t group;
  bitmask_iter_t bitmask;
} table_iterhash_t;

static FAST_PATH table_iterhash_t 
new_table_iterhash(table_t* table, uint64_t hash) {
  table_iterhash_t iter;
  iter.table = table;
  iter.h2_hash = h2(hash);
  iter.probe_seq = probe_seq(table, hash);
  iter.group = load(table->ctrl + iter.probe_seq.pos);
  iter.bitmask = match_byte(iter.group, iter.h2_hash);
  return iter;
}

static FAST_PATH bucket_t 
iterhash_next(table_iterhash_t* iter) {
  while (true) {
    size_t bit = next_mask(&iter->bitmask);
    if (bit != (size_t)-1) {
      size_t index = (iter->probe_seq.pos + bit) & iter->table->bucket_mask;
      return bucket(iter->table, index);
    }
    if (likely(any_bit_set(match_empty(iter->group)))) {
      return NULL;
    }
    move_next(&iter->probe_seq, iter->table->bucket_mask);
    iter->group = load(iter->table->ctrl + iter->probe_seq.pos);
    iter->bitmask = match_byte(iter->group, iter->h2_hash);
  }
}

static FAST_PATH bucket_t 
table_find(table_t* table, uint64_t hash, kk_box_t key, kk_context_t* ctx) {
  table_iterhash_t iter = new_table_iterhash(table, hash);
  bucket_t item = iterhash_next(&iter);
  while (item) {
    if (likely(apply_cmp(table->comparator, key, read_bucket(item), ctx))) {
      return item;
    }
    item = iterhash_next(&iter);
  }
  return NULL;
}

static FAST_PATH bool 
table_remove(table_t* table, uint64_t hash, kk_box_t key, kk_context_t* ctx) {
  bucket_t result = table_find(table, hash, key, ctx);
  if (result) {
    erase(table, result, ctx);
  }
  return result != NULL;
}

typedef struct kk_hashtable_s {
  struct kk_hashbrown__hashtable_s _base;
  kk_free_fun_t* free;
  kk_context_t* ctx; 
  table_t table;
} kk_hashtable_t;

static void kk_htable_free(void* ctx, kk_block_t* htable) {
  table_t *table = &((kk_hashtable_t*)htable)->table;
  clear(table, (kk_context_t*)ctx);
  free_buckets(table);
}

static kk_hashbrown__hashtable htable_create(kk_function_t hasher, kk_function_t comparator, kk_context_t* ctx) {
  kk_hashtable_t* htable =
    kk_block_alloc_as(kk_hashtable_t, 0, KK_TAG_CPTR_RAW, ctx);
  htable->table = new_table(hasher, comparator);
  htable->ctx = ctx; // TODO: maybe Koka can allow passing ctx to raw free path?
  htable->free = kk_htable_free;
  return kk_datatype_from_ptr(&htable->_base._block);
}

static kk_std_core__list kk_htable_to_list(kk_hashbrown__hashtable htable, kk_context_t* ctx) {
  table_t *table = &((kk_hashtable_t*)(htable.ptr))->table;
  table_iter_t iter = table_iterator(table);
  bucket_t item = next_table_item(&iter);
  kk_std_core__list list = kk_std_core__new_Nil(ctx);
  while (item) {
    kk_std_core__list hd = kk_std_core__new_Cons(kk_reuse_null,kk_box_dup(read_bucket(item)), list, ctx);
    item = next_table_item(&iter);
    list = hd;
  }
  kk_hashbrown__hashtable_drop(htable,ctx);
  return list;
}

static kk_unit_t
kk_htable_insert(kk_hashbrown__hashtable htable, kk_std_core_types__box data, kk_context_t* ctx) {
  table_t *table = &((kk_hashtable_t*)(htable.ptr))->table;
  size_t hash = apply_hash(table->hasher, data.unbox, ctx);
  insert(table, hash, data.unbox, ctx);
  kk_hashbrown__hashtable_drop(htable,ctx);
  return kk_Unit;
}

static bool
kk_htable_contains(kk_hashbrown__hashtable htable, kk_std_core_types__box key, kk_context_t* ctx) {
  table_t *table = &((kk_hashtable_t*)(htable.ptr))->table;
  size_t hash = apply_hash(table->hasher, key.unbox, ctx);
  bool flag = table_find(table, hash, key.unbox, ctx);
  kk_hashbrown__hashtable_drop(htable,ctx);
  return flag;
}

static bool
kk_htable_remove(kk_hashbrown__hashtable htable, kk_std_core_types__box key, kk_context_t* ctx) {
  table_t *table = &((kk_hashtable_t*)(htable.ptr))->table;
  size_t hash = apply_hash(table->hasher, key.unbox, ctx);
  bool flag = table_remove(table, hash, key.unbox, ctx);
  kk_hashbrown__hashtable_drop(htable,ctx);
  return flag;
}

static kk_integer_t
kk_htable_size(kk_hashbrown__hashtable htable, kk_context_t* ctx) {
  table_t *table = &((kk_hashtable_t*)(htable.ptr))->table;
  kk_integer_t res = kk_integer_from_uint64(table->items, ctx);
  kk_hashbrown__hashtable_drop(htable,ctx);
  return res;
}

static kk_integer_t
kk_htable_capacity(kk_hashbrown__hashtable htable, kk_context_t* ctx) {
  table_t *table = &((kk_hashtable_t*)(htable.ptr))->table;
  kk_integer_t res = kk_integer_from_uint64(table->items + table->growth_left, ctx);
  kk_hashbrown__hashtable_drop(htable,ctx);
  return res;
}

static kk_unit_t
kk_htable_shrink(kk_hashbrown__hashtable htable, size_t size, kk_context_t* ctx) {
  table_t *table = &((kk_hashtable_t*)(htable.ptr))->table;
  shrink_to(table, size, ctx);
  kk_hashbrown__hashtable_drop(htable,ctx);
  return kk_Unit;
}

static kk_unit_t
kk_htable_clear(kk_hashbrown__hashtable htable, kk_context_t* ctx) {
  table_t *table = &((kk_hashtable_t*)(htable.ptr))->table;
  clear(table, ctx);
  kk_hashbrown__hashtable_drop(htable,ctx);
  return kk_Unit;
}




/// NOTICE:
/// - boxes are dupped when insertion is determined
/// - boxes are dropped when erasure happened in inner structures
/// - therefore, in the interfaces, we should not dup/drop any box