import ahash

include extern {
  c  file "hashbrown.c"
}

type hashtable :: (H, V) -> V;

extern create-int-table: forall<h> (hasher: box<int> -> size_t, 
  comparator: (box<int>, box<int>) -> bool) -> hashtable<h, int> {
  // use box<a> as it is transparent as kk_box_t
  c inline "htable_create(#1, #2, kk_context())"
}

fun int-table(): ndet hashtable<h, int>  {
  val s = next-state();
  create-int-table(fn(x) { s.single(x.unbox) }, fn(x, y) { x.unbox == y.unbox })
}

fun seeded-int-table(seed: int32): ndet hashtable<h, int>  {
  val s = seed-state(seed);
  create-int-table(fn(x) { s.single(x.unbox) }, fn(x, y) { x.unbox == y.unbox })
}

extern list: forall<h, a> (table: hashtable<h, a>) -> read<h> list<a> {
  c inline "kk_htable_to_list(#1, kk_context())"
}

extern insert-boxed-int: forall<h, a> (table: hashtable<h, a>, value: box<int>) -> write<h> ()  {
  c inline "kk_htable_insert(#1, #2, kk_context())"
}

fun insert(table: hashtable<h, int>, x: int): write<h> hashtable<h, int> {
  insert-boxed-int(table, Box(x))
  table
}

extern contains-boxed-int(table: hashtable<h, int>, x: box<int>): read<h> bool {
  c inline "kk_htable_contains(#1, #2, kk_context())"
}

fun contains(table: hashtable<h, int>, x: int): read<h> bool {
  contains-boxed-int(table, Box(x))
}

extern remove-boxed-int(table: hashtable<h, int>, x: box<int>): <read<h>, write<h>> bool {
  c inline "kk_htable_remove(#1, #2, kk_context())"
}

fun remove(table: hashtable<h, int>, x: int): <read<h>, write<h>> (hashtable<h, int>, bool) {
  val res = remove-boxed-int(table, Box(x))
  (table, res)
}

