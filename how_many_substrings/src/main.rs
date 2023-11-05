use std::io;
use std::cmp::min;
use std::vec::Vec;
use std::ops::{Range, Add, AddAssign, Neg, Mul};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::fmt;

// extern crate algorithms;
// use algorithms::data_structures::{RBTree, FenwickTree, SegmentTree};
// use algorithms::string::generate_suffix_array_manber_myers;

fn lcp_construction(str_bytes: &[u8], suffix_array: &Vec<usize>) -> Vec<usize> {
    let n = str_bytes.len();
    let sa_lookup = inverse_permutation(&suffix_array);
    let mut prefix_array = vec![0; n];
    let mut prefix_counter: usize = 0;
    for suffix_start in 0..n {
        let sa_pos = sa_lookup[suffix_start];
        if sa_pos == n-1 {
            prefix_counter = 0;
            continue;
        }
        let sa_follower_start = suffix_array[sa_pos + 1];
        while
            suffix_start + prefix_counter < n &&
            sa_follower_start + prefix_counter < n &&
            str_bytes[suffix_start + prefix_counter] == str_bytes[sa_follower_start + prefix_counter]
        {
            prefix_counter += 1;
        }
        prefix_array[sa_pos + 1] = prefix_counter;
        if prefix_counter > 0 {
            prefix_counter -= 1;
        }
    }
    return prefix_array
}


fn inverse_permutation(permutation: &Vec<usize>) -> Vec<usize> {
    let n = permutation.len();
    let mut inverse_permutation: Vec<usize> = vec![0; n];
    for i in 0..n {
        inverse_permutation[permutation[i]] = i;
    }
    return inverse_permutation;
}


struct AdvanceFenwickTree {
    linear_fwt: FenwickTree<i32>,
    const_fwt: FenwickTree<i32>,
    len: usize,
}


impl AdvanceFenwickTree {
    pub fn add(&mut self, start: usize, end: usize, value: i32) {
        if start >= end {
            return;
        }
        self.linear_fwt.add(start, value);
        self.linear_fwt.add(end, -value);
        self.const_fwt.add(start, -value * (i32::try_from(start).unwrap() - 1));
        self.const_fwt.add(end, value * (i32::try_from(end).unwrap() - 1));
    }

    pub fn prefix_sum(&self, end: usize) -> i32 {
        if end == 0 {
            return 0;
        }
        let lin_part = self.linear_fwt.prefix_sum(end - 1) * (i32::try_from(end).unwrap() - 1);
        let const_part = self.const_fwt.prefix_sum(end - 1);
        return lin_part + const_part;
    }

    pub fn range_sum(&self, start: usize, end:usize) -> i32 {
        return self.prefix_sum(end) + (- self.prefix_sum(start));
    }

    pub fn with_len(len: usize) -> Self {
        AdvanceFenwickTree {
            linear_fwt: FenwickTree::with_len(len),
            const_fwt: FenwickTree::with_len(len),
            len: len
        }
    }
}

fn _update_prefix_counter_with(fw_counter: &mut AdvanceFenwickTree, left_len: usize, right_len: usize, comon_prefix: usize, value: i32) {
    let min_len = min(left_len, right_len);
    fw_counter.add(0, 1, value * i32::try_from(comon_prefix).unwrap());
    fw_counter.add(min_len - comon_prefix + 1, min_len + 1, -value);
}

fn add_prefix(fw_counter: &mut AdvanceFenwickTree, left_len: usize, right_len: usize, comon_prefix: usize) {
    _update_prefix_counter_with(fw_counter, left_len, right_len, comon_prefix, 1);
}

fn remove_prefix(fw_counter: &mut AdvanceFenwickTree, left_len: usize, right_len: usize, comon_prefix: usize) {
    _update_prefix_counter_with(fw_counter, left_len, right_len, comon_prefix, -1);
}


struct Query {
    start: usize,
    end: usize,
    id: usize,
}


impl fmt::Debug for AdvanceFenwickTree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut v = vec![0; self.len];
        for i in 1..self.len {
            v[i] = self.prefix_sum(i);
        }
        write!(f, "Advanced FW prefix sums: {v:?}")
    }
}


impl fmt::Debug for Query {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let s = self.start;
        let e = self.end;
        let id = self.id;
        write!(f, "Query (start: {s}, end: {e}, id: {id})")
    }
}


impl RBTree<usize, i8> {
    fn to_vec(&self) -> Vec<usize> {
        let mut v = vec![];
        let mut r: &RBNode<usize, i8>;
        unsafe {
            r = &(*self.root);
        }
        loop {
            let next = self.previous_by_key(r);
            if !next.is_none() {
                r = next.unwrap();
            } else {
                break;
            }
        }
        loop {
            v.push(r.key);
            let next = self.next_by_key(r);
            if next.is_none() {
                break;
            } else {
                r = next.unwrap();
            }
        }
        return v;
    }
}


impl fmt::Debug for RBTree<usize, i8> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let v = self.to_vec();
        write!(f, "RbTree {v:?}")
    }
}

struct Result {
    value: i32,
    id: usize,
}


fn count_substrings(str: String, raw_queries: Vec<Range<usize>>) -> Vec<i32> {
    let n = str.len();
    if n == 0 {
        return vec![];
    }

    let sa = generate_suffix_array_manber_myers(&str);
    let sa_lookup = inverse_permutation(&sa);
    let lcp = lcp_construction(str.as_bytes(), &sa);
    let min_lcp_segment_tree = SegmentTree::from_vec(&lcp, min);

    println!("{str}");
    println!("Sa: {sa:?}");
    println!("Lookup: {sa_lookup:?}");
    println!("Lcp: {lcp:?}");

    let mut suffix_sorter: RBTree<usize, i8> = RBTree::new();
    let mut fw_prefix_counter = AdvanceFenwickTree::with_len(n);

    let mut queries: Vec<Query> = Vec::with_capacity(raw_queries.len());
    for idx in 0..raw_queries.len() {
        let q = &raw_queries[idx];
        queries.push(Query{start: q.start, end: q.end, id: idx});
    }
    queries.sort_by(|a, b| a.start.cmp(&b.start));
    let mut results: Vec<Result> = Vec::with_capacity(raw_queries.len());

    dbg!(&queries);

    let mut query = queries.pop().unwrap();
    'main_loop: for suffix in (0..n).rev() {
        let sa_pos = sa_lookup[suffix];
        let x = &str;
        let y = &x[suffix..];
        println!("Substr: {y}");
        dbg!(&sa_pos);
        suffix_sorter.insert(sa_pos, 0);

        let _ss_vec = suffix_sorter.to_vec();
        println!("Suffix sorter: {_ss_vec:?}");
        let mut partial_sa: Vec<usize> = vec![];
        let mut partial_lcp: Vec<usize> = vec![];
        partial_sa.push(sa[_ss_vec[0]]);
        partial_lcp.push(0);
        for idx in 1.._ss_vec.len() {
            partial_sa.push(sa[_ss_vec[idx]]);
            partial_lcp.push(min_lcp_segment_tree.query(Range {start: _ss_vec[idx - 1] + 1, end: _ss_vec[idx] + 1 }).unwrap());
        }
        println!("Partial sa {partial_sa:?}");
        println!("Partial lcp {partial_lcp:?}");


        let node = suffix_sorter.find_node(&sa_pos).unwrap();

        let prev_node = suffix_sorter.previous_by_key(node);
        let next_node = suffix_sorter.next_by_key(node);

        if suffix == n-1 {
            continue;
        }

        if !prev_node.is_none() && !next_node.is_none() {
            let sa_left = prev_node.unwrap().key;
            let sa_right = next_node.unwrap().key;
            let cp = min_lcp_segment_tree.query(sa_left+1..sa_right+1).unwrap();

            dbg!(&sa_left);
            dbg!(&sa_right);
            dbg!(&cp);

            remove_prefix(
                &mut fw_prefix_counter,
                n - sa[sa_left],
                n - sa[sa_right],
                cp
            );
        }
        if !prev_node.is_none() {
            let sa_left = prev_node.unwrap().key;
            let cp = min_lcp_segment_tree.query(sa_left+1..sa_pos+1).unwrap();

            dbg!(&sa_left);
            dbg!(&cp);

            add_prefix(
                &mut fw_prefix_counter,
                n - sa[sa_left],
                n - sa[sa_pos],
                cp
            );
        }
        if !next_node.is_none() {
            let sa_right = next_node.unwrap().key;
            let cp = min_lcp_segment_tree.query(sa_pos+1..sa_right+1).unwrap();

            dbg!(&sa_right);
            dbg!(&cp);

            add_prefix(
                &mut fw_prefix_counter,
                n - sa[sa_pos],
                n - sa[sa_right],
                cp
            );
        }
        dbg!(&fw_prefix_counter);

        while query.start == suffix {
            let max_substrings = i32::try_from((query.end - query.start) * (query.end - query.start + 1) / 2).unwrap();
            let res = max_substrings - fw_prefix_counter.prefix_sum(n - query.end + 1);
            results.push(Result { value: res, id: query.id });
            match queries.pop() {
                Some(q) => query = q,
                None => break 'main_loop,
            };
        }
    }
    results.sort_by_key(|r| r.id);

    let mut output: Vec<i32> = Vec::with_capacity(results.len());
    for r in results {
        output.push(r.value);
    }
    return output;
}


fn parse_line_to_two_numbers() -> (i32, i32) {
    let mut line = String::new();
    io::stdin().read_line(&mut line).unwrap();
    let _tuple_sizes: Vec<&str> = line.split(" ").collect();
    let a = _tuple_sizes[0].trim().parse::<i32>().unwrap();
    let b = _tuple_sizes[1].trim().parse::<i32>().unwrap();
    return (a, b);
}


fn main() {
    let (_, q) = parse_line_to_two_numbers();
    let mut str = String::new();
    io::stdin().read_line(&mut str).unwrap();
    str = (str.trim()).to_string();

    let mut queries: Vec<Range<usize>>= vec![];
    for _ in 0..q {
        let (start, end) = parse_line_to_two_numbers();
        queries.push(Range {start: usize::try_from(start).unwrap(), end: usize::try_from(end).unwrap() + 1});
    }

    let results = count_substrings(str, queries);

    for r in results {
        println!("{r}");
    }
}


#[cfg(test)]
mod tests {
    use super::count_substrings;
    use super::SegmentTree;
    use super::min;
    use std::ops::Range;

    #[test]
    fn segment_tree_test() {
        let v: Vec<usize> = vec![11, 12, 13, 24, 0, 16, 7, 8];
        let s: SegmentTree<usize> = SegmentTree::from_vec(&v, min);
        let queries: Vec<Range<usize>> = vec![
            Range{start: 0, end: 4},
            Range{start: 2, end: 5},
            Range{start: 4, end: 8},
            Range{start: 5, end: 6},
        ];
        let res: Vec<usize>= vec![11, 0, 0, 16];
        for i in 0..res.len() {
            let x = &queries[i];
            let y = s.query(Range { start: (x.start), end: (x.end) }).unwrap();
            assert_eq!(y, res[i]);
        }
    }

    #[test]
    fn example1() {
        let str = String::from("aaabbabaaa");
        let queries = vec![
            Range{start: 8, end: 10},
            Range{start: 7, end: 9},
            Range{start: 5, end: 9},
            Range{start: 2, end: 5},
            Range{start: 1, end: 9},
        ];
        let expected = vec![2, 2, 8, 5, 27];
        let results = count_substrings(str, queries);

        for idx in 0..results.len() {
            assert_eq!(results[idx], expected[idx]);
        }
    }

    #[test]
    fn example2() {
        let str = String::from("aabaa");
        let queries = vec![
            Range{start: 1, end: 2},
            Range{start: 1, end: 5},
            Range{start: 1, end: 2},
            Range{start: 1, end: 5},
            Range{start: 0, end: 3},
        ];
        let expected = vec![1, 8, 1, 8, 5];
        let results = count_substrings(str, queries);

        for idx in 0..results.len() {
            assert_eq!(results[idx], expected[idx]);
        }
    }

    // #[test]
    // fn example3() {
    //     let str = String::from(
    //         "ccccccccccgdgddgdgddgddgdgddgdgddgddgdgddgddgdgddgdgddgddgdgddwbwbbwbwbbwbbwbwbbwbkgkggwyomdjdbevunm"
    //     );
    //     let queries = vec![
    //         Range{start: 62, end: 70},
    //     ];

    //     let expected = vec![24];
    //     let results = count_substrings(str, queries);

    //     for idx in 0..results.len() {
    //         assert_eq!(results[idx], expected[idx]);
    //     }
    // }

    #[test]
    fn example_tmp() {
        let str = String::from(
            "wbbwbwbbwbbw"
        );
        let queries = vec![
            Range{start: 7, end: 12},
            Range{start: 0, end: 8},
        ];

        let expected = vec![11, 24];
        let results = count_substrings(str, queries);

        for idx in 0..results.len() {
            assert_eq!(results[idx], expected[idx]);
        }
    }

    #[test]
    fn example_tmp2() {
        let str = String::from(
            "bwbbw"
        );
        let queries = vec![
            Range{start: 0, end: 3},
        ];

        let expected = vec![5];
        let results = count_substrings(str, queries);

        for idx in 0..results.len() {
            assert_eq!(results[idx], expected[idx]);
        }
    }

    // #[test]
    // fn example_tmp2() {
    //     let str = String::from(
    //         "wbwbbwbwbbwbbwbwbb"
    //     );
    //     let queries = vec![
    //         Range{start: 0, end: 8},
    //     ];

    //     let expected = vec![24];
    //     let results = count_substrings(str, queries);

    //     for idx in 0..results.len() {
    //         assert_eq!(results[idx], expected[idx]);
    //     }
    // }
}





// ## COPY OF HELPER STRUCTURES -> to make the script self-sufficient


/// Fenwick Tree / Binary Indexed Tree
///
/// Consider we have an array `arr[0...n-1]`. We would like to:
/// 1. Compute the sum of the first i elements.
/// 2. Modify the value of a specified element of the array `arr[i] = x`, where `0 <= i <= n-1`.
pub struct FenwickTree<T: Add + AddAssign + Copy + Default> {
    data: Vec<T>,
}

impl<T: Add<Output = T> + AddAssign + Copy + Default> FenwickTree<T> {
    /// construct a new FenwickTree with given length
    pub fn with_len(len: usize) -> Self {
        FenwickTree {
            data: vec![T::default(); len + 1],
        }
    }

    /// add `val` to `idx`
    pub fn add(&mut self, i: usize, val: T) {
        assert!(i < self.data.len());
        let mut i = i + 1;
        while i < self.data.len() {
            self.data[i] += val;
            i += lowbit(i);
        }
    }

    /// get the sum of [0, i]
    pub fn prefix_sum(&self, i: usize) -> T {
        assert!(i < self.data.len());
        let mut i = i + 1;
        let mut res = T::default();
        while i > 0 {
            res += self.data[i];
            i -= lowbit(i);
        }
        res
    }
}

/// get the lowest bit of `i`
const fn lowbit(x: usize) -> usize {
    let x = x as isize;
    (x & (-x)) as usize
}

use std::boxed::Box;
use std::cmp::{Ord, Ordering};
use std::iter::Iterator;
use std::ptr::null_mut;

#[derive(Copy, Clone)]
enum Color {
    Red,
    Black,
}

pub struct RBNode<K: Ord, V> {
    pub key: K,
    pub value: V,
    color: Color,
    pub parent: *mut RBNode<K, V>,
    pub left: *mut RBNode<K, V>,
    pub right: *mut RBNode<K, V>,
}

impl<K: Ord, V> RBNode<K, V> {
    fn new(key: K, value: V) -> RBNode<K, V> {
        RBNode {
            key,
            value,
            color: Color::Red,
            parent: null_mut(),
            left: null_mut(),
            right: null_mut(),
        }
    }
}

pub struct RBTree<K: Ord, V> {
    root: *mut RBNode<K, V>,
}

impl<K: Ord, V> Default for RBTree<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, V> RBTree<K, V> {
    pub fn new() -> RBTree<K, V> {
        RBTree::<K, V> { root: null_mut() }
    }

    pub fn find(&self, key: &K) -> Option<&V> {
        let option = self.find_node(key);
        return match option {
            None => None,
            Some(node) => Some(&node.value),
        };
    }

    pub fn find_node(&self, key: &K) -> Option<&RBNode<K, V>> {
        unsafe {
            let mut node = self.root;
            while !node.is_null() {
                node = match (*node).key.cmp(key) {
                    Ordering::Less => (*node).right,
                    Ordering::Equal => return Some(&(*node)),
                    Ordering::Greater => (*node).left,
                }
            }
        }
        None
    }

    pub fn next_by_key(&self, node: &RBNode<K, V>) -> Option<&RBNode<K, V>> {
        unsafe {
            let mut x: *const RBNode<K, V> = &(*node);
            if !(*x).right.is_null() {
                x = (*x).right;
                while !(*x).left.is_null() {
                    x = (*x).left;
                }
                return Some(&(*x));
            } else {
                while !(*x).parent.is_null() {
                    let p = (*x).parent;
                    if x == (*p).left {
                        return Some(&(*p));
                    }
                    x = p;
                }
                return None
            }
        }
    }

    pub fn previous_by_key(&self, node: &RBNode<K, V>) -> Option<&RBNode<K, V>> {
        unsafe {
            let mut x: *const RBNode<K, V> = &(*node);
            if !(*x).left.is_null() {
                x = (*x).left;
                while !(*x).right.is_null() {
                    x = (*x).right;
                }
                return Some(&(*x));
            } else {
                while !(*x).parent.is_null() {
                    let p = (*x).parent;
                    if x == (*p).right{
                        return Some(&(*p));
                    }
                    x = p;
                }
                return None
            }
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        unsafe {
            let mut parent = null_mut();
            let mut node = self.root;
            while !node.is_null() {
                parent = node;
                node = match (*node).key.cmp(&key) {
                    Ordering::Less => (*node).right,
                    Ordering::Equal => {
                        (*node).value = value;
                        return;
                    }
                    Ordering::Greater => (*node).left,
                }
            }
            node = Box::into_raw(Box::new(RBNode::new(key, value)));
            if !parent.is_null() {
                if (*node).key < (*parent).key {
                    (*parent).left = node;
                } else {
                    (*parent).right = node;
                }
            } else {
                self.root = node;
            }
            (*node).parent = parent;
            insert_fixup(self, node);
        }
    }

    pub fn delete(&mut self, key: &K) {
        unsafe {
            let mut parent = null_mut();
            let mut node = self.root;
            while !node.is_null() {
                node = match (*node).key.cmp(key) {
                    Ordering::Less => {
                        parent = node;
                        (*node).right
                    }
                    Ordering::Equal => break,
                    Ordering::Greater => {
                        parent = node;
                        (*node).left
                    }
                };
            }

            if node.is_null() {
                return;
            }

            /* cl and cr denote left and right child of node, respectively. */
            let cl = (*node).left;
            let cr = (*node).right;
            let mut deleted_color;

            if cl.is_null() {
                replace_node(self, parent, node, cr);
                if cr.is_null() {
                    /*
                     * Case 1 - cl and cr are both NULL
                     * (n could be either color here)
                     *
                     *     (n)             NULL
                     *    /   \    -->
                     *  NULL  NULL
                     */

                    deleted_color = (*node).color;
                } else {
                    /*
                     * Case 2 - cl is NULL and cr is not NULL
                     *
                     *     N             Cr
                     *    / \    -->    /  \
                     *  NULL cr       NULL NULL
                     */

                    (*cr).parent = parent;
                    (*cr).color = Color::Black;
                    deleted_color = Color::Red;
                }
            } else if cr.is_null() {
                /*
                 * Case 3 - cl is not NULL and cr is NULL
                 *
                 *     N             Cl
                 *    / \    -->    /  \
                 *  cl  NULL      NULL NULL
                 */

                replace_node(self, parent, node, cl);
                (*cl).parent = parent;
                (*cl).color = Color::Black;
                deleted_color = Color::Red;
            } else {
                let mut victim = (*node).right;
                while !(*victim).left.is_null() {
                    victim = (*victim).left;
                }
                if victim == (*node).right {
                    /* Case 4 - victim is the right child of node
                     *
                     *     N         N         n
                     *    / \       / \       / \
                     *  (cl) cr   (cl) Cr    Cl  Cr
                     *
                     *     N         n
                     *    / \       / \
                     *  (cl) Cr    Cl  Cr
                     *         \         \
                     *         crr       crr
                     */

                    replace_node(self, parent, node, victim);
                    (*victim).parent = parent;
                    deleted_color = (*victim).color;
                    (*victim).color = (*node).color;
                    (*victim).left = cl;
                    (*cl).parent = victim;
                    if (*victim).right.is_null() {
                        parent = victim;
                    } else {
                        deleted_color = Color::Red;
                        (*(*victim).right).color = Color::Black;
                    }
                } else {
                    /*
                     * Case 5 - victim is not the right child of node
                     */

                    /* vp and vr denote parent and right child of victim, respectively. */
                    let vp = (*victim).parent;
                    let vr = (*victim).right;
                    (*vp).left = vr;
                    if vr.is_null() {
                        deleted_color = (*victim).color;
                    } else {
                        deleted_color = Color::Red;
                        (*vr).parent = vp;
                        (*vr).color = Color::Black;
                    }
                    replace_node(self, parent, node, victim);
                    (*victim).parent = parent;
                    (*victim).color = (*node).color;
                    (*victim).left = cl;
                    (*victim).right = cr;
                    (*cl).parent = victim;
                    (*cr).parent = victim;
                    parent = vp;
                }
            }

            /* release resource */
            drop(Box::from_raw(node));
            if matches!(deleted_color, Color::Black) {
                delete_fixup(self, parent);
            }
        }
    }

    pub fn iter<'a>(&self) -> RBTreeIterator<'a, K, V> {
        let mut iterator = RBTreeIterator { stack: Vec::new() };
        let mut node = self.root;
        unsafe {
            while !node.is_null() {
                iterator.stack.push(&*node);
                node = (*node).left;
            }
        }
        iterator
    }
}

#[inline]
unsafe fn insert_fixup<K: Ord, V>(tree: &mut RBTree<K, V>, mut node: *mut RBNode<K, V>) {
    let mut parent: *mut RBNode<K, V> = (*node).parent;
    let mut gparent: *mut RBNode<K, V>;
    let mut tmp: *mut RBNode<K, V>;

    loop {
        /*
         * Loop invariant:
         * - node is red
         */

        if parent.is_null() {
            (*node).color = Color::Black;
            break;
        }

        if matches!((*parent).color, Color::Black) {
            break;
        }

        gparent = (*parent).parent;
        tmp = (*gparent).right;
        if parent != tmp {
            /* parent = (*gparent).left */
            if !tmp.is_null() && matches!((*tmp).color, Color::Red) {
                /*
                 * Case 1 - color flips and recurse at g
                 *
                 *      G               g
                 *     / \             / \
                 *    p   u    -->    P   U
                 *   /               /
                 *  n               n
                 */

                (*parent).color = Color::Black;
                (*tmp).color = Color::Black;
                (*gparent).color = Color::Red;
                node = gparent;
                parent = (*node).parent;
                continue;
            }
            tmp = (*parent).right;
            if node == tmp {
                /* node = (*parent).right */
                /*
                 * Case 2 - left rotate at p (then Case 3)
                 *
                 *    G               G
                 *   / \             / \
                 *  p   U    -->    n   U
                 *   \             /
                 *    n           p
                 */

                left_rotate(tree, parent);
                parent = node;
            }
            /*
             * Case 3 - right rotate at g
             *
             *      G               P
             *     / \             / \
             *    p   U    -->    n   g
             *   /                     \
             *  n                       U
             */

            (*parent).color = Color::Black;
            (*gparent).color = Color::Red;
            right_rotate(tree, gparent);
        } else {
            /* parent = (*gparent).right */
            tmp = (*gparent).left;
            if !tmp.is_null() && matches!((*tmp).color, Color::Red) {
                /*
                 * Case 1 - color flips and recurse at g
                 *    G               g
                 *   / \             / \
                 *  u   p    -->    U   P
                 *       \               \
                 *        n               n
                 */

                (*parent).color = Color::Black;
                (*tmp).color = Color::Black;
                (*gparent).color = Color::Red;
                node = gparent;
                parent = (*node).parent;
                continue;
            }
            tmp = (*parent).left;
            if node == tmp {
                /*
                 * Case 2 - right rotate at p (then Case 3)
                 *
                 *       G             G
                 *      / \           / \
                 *     U   p   -->   U   n
                 *        /               \
                 *       n                 p
                 */

                right_rotate(tree, parent);
                parent = node;
            }
            /*
             * Case 3 - left rotate at g
             *
             *       G             P
             *      / \           / \
             *     U   p   -->   g   n
             *          \       /
             *           n     U
             */

            (*parent).color = Color::Black;
            (*gparent).color = Color::Red;
            left_rotate(tree, gparent);
        }
        break;
    }
}

#[inline]
unsafe fn delete_fixup<K: Ord, V>(tree: &mut RBTree<K, V>, mut parent: *mut RBNode<K, V>) {
    let mut node: *mut RBNode<K, V> = null_mut();
    let mut sibling: *mut RBNode<K, V>;
    /* sl and sr denote left and right child of sibling, respectively. */
    let mut sl: *mut RBNode<K, V>;
    let mut sr: *mut RBNode<K, V>;

    loop {
        /*
         * Loop invariants:
         * - node is black (or null on first iteration)
         * - node is not the root (so parent is not null)
         * - All leaf paths going through parent and node have a
         *   black node count that is 1 lower than other leaf paths.
         */
        sibling = (*parent).right;
        if node != sibling {
            /* node = (*parent).left */
            if matches!((*sibling).color, Color::Red) {
                /*
                 * Case 1 - left rotate at parent
                 *
                 *    P               S
                 *   / \             / \
                 *  N   s    -->    p   Sr
                 *     / \         / \
                 *    Sl  Sr      N  Sl
                 */

                left_rotate(tree, parent);
                (*parent).color = Color::Red;
                (*sibling).color = Color::Black;
                sibling = (*parent).right;
            }
            sl = (*sibling).left;
            sr = (*sibling).right;

            if !sl.is_null() && matches!((*sl).color, Color::Red) {
                /*
                 * Case 2 - right rotate at sibling and then left rotate at parent
                 * (p and sr could be either color here)
                 *
                 *   (p)             (p)              (sl)
                 *   / \             / \              / \
                 *  N   S    -->    N   sl    -->    P   S
                 *     / \                \         /     \
                 *    sl (sr)              S       N      (sr)
                 *                          \
                 *                          (sr)
                 */

                (*sl).color = (*parent).color;
                (*parent).color = Color::Black;
                right_rotate(tree, sibling);
                left_rotate(tree, parent);
            } else if !sr.is_null() && matches!((*sr).color, Color::Red) {
                /*
                 * Case 3 - left rotate at parent
                 * (p could be either color here)
                 *
                 *   (p)               S
                 *   / \              / \
                 *  N   S    -->    (p) (sr)
                 *     / \          / \
                 *    Sl  sr       N   Sl
                 */

                (*sr).color = (*parent).color;
                left_rotate(tree, parent);
            } else {
                /*
                 * Case 4 - color clip
                 * (p could be either color here)
                 *
                 *   (p)             (p)
                 *   / \             / \
                 *  N   S    -->    N   s
                 *     / \             / \
                 *    Sl  Sr          Sl  Sr
                 */

                (*sibling).color = Color::Red;
                if matches!((*parent).color, Color::Black) {
                    node = parent;
                    parent = (*node).parent;
                    continue;
                }
                (*parent).color = Color::Black;
            }
        } else {
            /* node = (*parent).right */
            sibling = (*parent).left;
            if matches!((*sibling).color, Color::Red) {
                /*
                 * Case 1 - right rotate at parent
                 */

                right_rotate(tree, parent);
                (*parent).color = Color::Red;
                (*sibling).color = Color::Black;
                sibling = (*parent).right;
            }
            sl = (*sibling).left;
            sr = (*sibling).right;

            if !sr.is_null() && matches!((*sr).color, Color::Red) {
                /*
                 * Case 2 - left rotate at sibling and then right rotate at parent
                 */

                (*sr).color = (*parent).color;
                (*parent).color = Color::Black;
                left_rotate(tree, sibling);
                right_rotate(tree, parent);
            } else if !sl.is_null() && matches!((*sl).color, Color::Red) {
                /*
                 * Case 3 - right rotate at parent
                 */

                (*sl).color = (*parent).color;
                right_rotate(tree, parent);
            } else {
                /*
                 * Case 4 - color flip
                 */

                (*sibling).color = Color::Red;
                if matches!((*parent).color, Color::Black) {
                    node = parent;
                    parent = (*node).parent;
                    continue;
                }
                (*parent).color = Color::Black;
            }
        }
        break;
    }
}

#[inline]
unsafe fn left_rotate<K: Ord, V>(tree: &mut RBTree<K, V>, x: *mut RBNode<K, V>) {
    /*
     * Left rotate at x
     * (x could also be the left child of p)
     *
     *  p           p
     *   \           \
     *    x    -->    y
     *   / \         / \
     *      y       x
     *     / \     / \
     *    c           c
     */

    let p = (*x).parent;
    let y = (*x).right;
    let c = (*y).left;

    (*y).left = x;
    (*x).parent = y;
    (*x).right = c;
    if !c.is_null() {
        (*c).parent = x;
    }
    if p.is_null() {
        tree.root = y;
    } else if (*p).left == x {
        (*p).left = y;
    } else {
        (*p).right = y;
    }
    (*y).parent = p;
}

#[inline]
unsafe fn right_rotate<K: Ord, V>(tree: &mut RBTree<K, V>, x: *mut RBNode<K, V>) {
    /*
     * Right rotate at x
     * (x could also be the left child of p)
     *
     *  p           p
     *   \           \
     *    x    -->    y
     *   / \         / \
     *  y               x
     * / \             / \
     *    c           c
     */

    let p = (*x).parent;
    let y = (*x).left;
    let c = (*y).right;

    (*y).right = x;
    (*x).parent = y;
    (*x).left = c;
    if !c.is_null() {
        (*c).parent = x;
    }
    if p.is_null() {
        tree.root = y;
    } else if (*p).left == x {
        (*p).left = y;
    } else {
        (*p).right = y;
    }
    (*y).parent = p;
}

#[inline]
unsafe fn replace_node<K: Ord, V>(
    tree: &mut RBTree<K, V>,
    parent: *mut RBNode<K, V>,
    node: *mut RBNode<K, V>,
    new: *mut RBNode<K, V>,
) {
    if parent.is_null() {
        tree.root = new;
    } else if (*parent).left == node {
        (*parent).left = new;
    } else {
        (*parent).right = new;
    }
}

pub struct RBTreeIterator<'a, K: Ord, V> {
    stack: Vec<&'a RBNode<K, V>>,
}

impl<'a, K: Ord, V> Iterator for RBTreeIterator<'a, K, V> {
    type Item = &'a RBNode<K, V>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.stack.pop() {
            Some(node) => {
                let mut next = node.right;
                unsafe {
                    while !next.is_null() {
                        self.stack.push(&*next);
                        next = (*next).left;
                    }
                }
                Some(node)
            }
            None => None,
        }
    }
}



/// This data structure implements a segment-tree that can efficiently answer range (interval) queries on arrays.
/// It represents this array as a binary tree of merged intervals. From top to bottom: [aggregated value for the overall array], then [left-hand half, right hand half], etc. until [each individual value, ...]
/// It is generic over a reduction function for each segment or interval: basically, to describe how we merge two intervals together.
/// Note that this function should be commutative and associative
///     It could be `std::cmp::min(interval_1, interval_2)` or `std::cmp::max(interval_1, interval_2)`, or `|a, b| a + b`, `|a, b| a * b`
pub struct SegmentTree<T: Debug + Default + Ord + Copy> {
    len: usize,           // length of the represented
    tree: Vec<T>, // represents a binary tree of intervals as an array (as a BinaryHeap does, for instance)
    merge: fn(T, T) -> T, // how we merge two values together
}

impl<T: Debug + Default + Ord + Copy> SegmentTree<T> {
    /// Builds a SegmentTree from an array and a merge function
    pub fn from_vec(arr: &[T], merge: fn(T, T) -> T) -> Self {
        let len = arr.len();
        let mut buf: Vec<T> = vec![T::default(); 2 * len];
        // Populate the tree bottom-up, from right to left
        buf[len..(2 * len)].clone_from_slice(&arr[0..len]); // last len pos is the bottom of the tree -> every individual value
        for i in (1..len).rev() {
            // a nice property of this "flat" representation of a tree: the parent of an element at index i is located at index i/2
            buf[i] = merge(buf[2 * i], buf[2 * i + 1]);
        }
        SegmentTree {
            len,
            tree: buf,
            merge,
        }
    }

    /// Query the range (exclusive)
    /// returns None if the range is out of the array's boundaries (eg: if start is after the end of the array, or start > end, etc.)
    /// return the aggregate of values over this range otherwise
    pub fn query(&self, range: Range<usize>) -> Option<T> {
        let mut l = range.start + self.len;
        let mut r = min(self.len, range.end) + self.len;
        let mut res = None;
        // Check Wikipedia or other detailed explanations here for how to navigate the tree bottom-up to limit the number of operations
        while l < r {
            if l % 2 == 1 {
                res = Some(match res {
                    None => self.tree[l],
                    Some(old) => (self.merge)(old, self.tree[l]),
                });
                l += 1;
            }
            if r % 2 == 1 {
                r -= 1;
                res = Some(match res {
                    None => self.tree[r],
                    Some(old) => (self.merge)(old, self.tree[r]),
                });
            }
            l /= 2;
            r /= 2;
        }
        res
    }

    /// Updates the value at index `idx` in the original array with a new value `val`
    pub fn update(&mut self, idx: usize, val: T) {
        // change every value where `idx` plays a role, bottom -> up
        // 1: change in the right-hand side of the tree (bottom row)
        let mut idx = idx + self.len;
        self.tree[idx] = val;

        // 2: then bubble up
        idx /= 2;
        while idx != 0 {
            self.tree[idx] = (self.merge)(self.tree[2 * idx], self.tree[2 * idx + 1]);
            idx /= 2;
        }
    }
}


pub fn generate_suffix_array_manber_myers(input: &str) -> Vec<usize> {
    if input.is_empty() {
        return Vec::new();
    }
    let n = input.len();
    let mut suffixes: Vec<(usize, &str)> = Vec::with_capacity(n);

    for (i, _suffix) in input.char_indices() {
        suffixes.push((i, &input[i..]));
    }

    suffixes.sort_by_key(|&(_, s)| s);

    let mut suffix_array: Vec<usize> = vec![0; n];
    let mut rank = vec![0; n];

    let mut cur_rank = 0;
    let mut prev_suffix = &suffixes[0].1;

    for (i, suffix) in suffixes.iter().enumerate() {
        if &suffix.1 != prev_suffix {
            cur_rank += 1;
            prev_suffix = &suffix.1;
        }
        rank[suffix.0] = cur_rank;
        suffix_array[i] = suffix.0;
    }

    let mut k = 1;
    let mut new_rank: Vec<usize> = vec![0; n];

    while k < n {
        suffix_array.sort_by_key(|&x| (rank[x], rank[(x + k) % n]));

        let mut cur_rank = 0;
        let mut prev = suffix_array[0];
        new_rank[prev] = cur_rank;

        for &suffix in suffix_array.iter().skip(1) {
            let next = suffix;
            if (rank[prev], rank[(prev + k) % n]) != (rank[next], rank[(next + k) % n]) {
                cur_rank += 1;
            }
            new_rank[next] = cur_rank;
            prev = next;
        }

        std::mem::swap(&mut rank, &mut new_rank);

        k <<= 1;
    }

    suffix_array
}
