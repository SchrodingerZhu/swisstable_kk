import ahash

include extern {
  c  file "hashbrown.c"
}

type hashtable<a>;

effect hashable<a> {
  fun hash( x : a ) : size_t
}

extern create-int-tablex(h: a -> size_t): hashtable<int> {
  c inline "htable_with_hasher(#1, kk_context())"
}

extern simple(x: a): size_t {
  c inline "0"
}

fun test-table(): hashtable<int>  {
  val k = 1;
  create-int-tablex(fn(x) {k.size_t})
}


