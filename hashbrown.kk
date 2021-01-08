import ahash

include extern {
  c  file "hashbrown.c"
}

type hashtable :: (H, V) -> V;

extern create-int-table: forall<h> (hasher: box<int> -> size_t) -> hashtable<h, int> {
  // use box<a> as it is transparent as kk_box_t
  c inline "htable_with_hasher(#1, kk_context())"
}

fun int-table(): ndet hashtable<h, int>  {
  val s = next-state();
  create-int-table(fn(x) { s.single(x.unbox) })
}

extern list: forall<h, a> (table: hashtable<h, a>) -> read<h> list<a> {
  c inline "kk_htable_to_list(#1, kk_context())"
}

extern insert-boxed-int: forall<h, a> (table: hashtable<h, a>, value: box<int>) -> write<h> ()  {
  c inline "kk_htable_insert(#1, #2, kk_context())"
}

fun insert(table: hashtable<h, a>, x: int): write<h> hashtable<h, a> {
  insert-boxed-int(table, Box(x))
  table
}

