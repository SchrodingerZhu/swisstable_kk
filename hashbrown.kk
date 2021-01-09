import ahash

include extern {
  c  file "hashbrown.c"
}

type hashtable :: (H, V) -> V;

extern create-int-tablex: forall<h> (hasher: box<int> -> size_t, 
  comparator: (box<int>, box<int>) -> bool) -> hashtable<h, int> {
  // use box<a> as it is transparent as kk_box_t
  c inline "htable_create(#1, #2, kk_context())"
}

fun create-int-table(): ndet hashtable<h, int>  {
  val s = next-state();
  create-int-tablex(fn(x) { s.single(x.unbox) }, fn(x, y) { x.unbox == y.unbox })
}

fun seeded-int-table(seed: int32): ndet hashtable<h, int>  {
  val s = seed-state(seed);
  create-int-tablex(fn(x) { s.single(x.unbox) }, fn(x, y) { x.unbox == y.unbox })
}

extern create-str-tablex: forall<h> (hasher: box<string> -> size_t, 
  comparator: (box<string>, box<string>) -> bool) -> hashtable<h, string> {
  // use box<a> as it is transparent as kk_box_t
  c inline "htable_create(#1, #2, kk_context())"
}

fun create-str-table(): ndet hashtable<h, string>  {
  val s = next-state();
  create-str-tablex(fn(x) { s.single(x.unbox) }, fn(x, y) { x.unbox == y.unbox })
}

fun seeded-str-table(seed: int32): ndet hashtable<h, string>  {
  val s = seed-state(seed);
  create-str-tablex(fn(x) { s.single(x.unbox) }, fn(x, y) { x.unbox == y.unbox })
}

extern list: forall<h, a> (table: hashtable<h, a>) -> read<h> list<a> {
  c inline "kk_htable_to_list(#1, kk_context())"
}

extern insert-boxed: forall<h, a> (table: hashtable<h, a>, value: box<a>) -> write<h> ()  {
  c inline "kk_htable_insert(#1, #2, kk_context())"
}

fun insert(table: hashtable<h, a>, x: a): write<h> hashtable<h, a> {
  insert-boxed(table, Box(x))
  table
}

extern contains-boxed(table: hashtable<h, a>, x: box<a>): read<h> bool {
  c inline "kk_htable_contains(#1, #2, kk_context())"
}

fun contains(table: hashtable<h, a>, x: a): read<h> bool {
  contains-boxed(table, Box(x))
}

extern remove-boxed(table: hashtable<h, a>, x: box<a>): <read<h>, write<h>> bool {
  c inline "kk_htable_remove(#1, #2, kk_context())"
}

fun remove(table: hashtable<h, a>, x: a): <read<h>, write<h>> (hashtable<h, a>, bool) {
  val res = remove-boxed(table, Box(x))
  (table, res)
}

