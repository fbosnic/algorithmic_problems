use std::io;
use std::cmp::{min,max};
use std::vec::Vec;
use std::ops::{Range, Add, AddAssign};
use std::convert::TryFrom;
use std::fmt::Debug;

// Hackerrank does not allow importing external packages
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


struct FWTreeWithRangedUpdates {
    linear_fwt: FenwickTree<i64>,
    const_fwt: FenwickTree<i64>
}


impl FWTreeWithRangedUpdates {
    pub fn add(&mut self, start: usize, end: usize, value: i64) {
        if start >= end {
            return;
        }
        self.linear_fwt.add(start, value);
        self.linear_fwt.add(end, -value);
        self.const_fwt.add(start, -value * (i64::try_from(start).unwrap() - 1));
        self.const_fwt.add(end, value * (i64::try_from(end).unwrap() - 1));
    }

    pub fn prefix_sum(&self, end: usize) -> i64 {
        if end == 0 {
            return 0;
        }
        let lin_part = self.linear_fwt.prefix_sum(end - 1) * (i64::try_from(end).unwrap() - 1);
        let const_part = self.const_fwt.prefix_sum(end - 1);
        return lin_part + const_part;
    }

    pub fn range_sum(&self, start: usize, end:usize) -> i64 {
        return self.prefix_sum(end) + (- self.prefix_sum(start));
    }

    pub fn with_len(len: usize) -> Self {
        FWTreeWithRangedUpdates {
            linear_fwt: FenwickTree::with_len(len),
            const_fwt: FenwickTree::with_len(len)
        }
    }
}


struct Query {
    start: usize,
    end: usize,
    id: usize,
}

struct Result {
    value: i64,
    id: usize,
}

fn find_distinguished_elements(
    suffix_idx: usize,
    lcp_seg_tree: &SegmentTree<usize>,
    sa_seg_tree: &SegmentTree<usize>,
    sa_lookup: &Vec<usize>,
) -> Vec<usize> {
    if suffix_idx == sa_lookup.len() - 1 {
        return vec![]
    }
    let pos = sa_lookup[suffix_idx];
    let mut dist_elements:  Vec<usize> = Vec::with_capacity(sa_lookup.len());
    let mut lcp_depth: usize = 0;
    loop {
        let mut left = find_left_limit(
            lcp_seg_tree,
            pos,
            lcp_depth,
        );
        left = max(left, 1);  // The first element in LCP is undefined
        let mut right = find_right_limit(
            lcp_seg_tree,
            pos,
            lcp_depth
        );
        right = min(right, sa_lookup.len());  // Segment trees arrays are enlarged to fit the binary tree
        if left > right {
            break;
        }
        let k = sa_seg_tree.query(Range{ start:left - 1, end:right + 1 }).expect("Should never happen");
        if k >= sa_lookup.len() {  // The segment array is not filled at these positions
            break;
        }
        dist_elements.push(k);
        let a = min(pos, sa_lookup[k]);
        let b = max(pos, sa_lookup[k]);
        lcp_depth = 1 + lcp_seg_tree.query(Range { start: a + 1, end: b + 1}).expect("Should not happen");
    }
    return dist_elements
}

fn get_node_from_index(idx: usize, array_len: usize) -> usize {
    return idx + array_len
}

fn get_index_from_node(node: usize, array_len: usize) -> usize {
    return node - array_len;
}

fn is_root(node: usize) -> bool {
    return node == 1;
}

fn parent(node: usize) -> usize {
    return node / 2;
}

fn left_child(node: usize) -> usize {
    return node * 2;
}

fn is_left_child(node: usize) -> bool {
    return node % 2 == 0;
}

fn right_child(node: usize) -> usize {
    return node * 2 + 1;
}

fn is_right_child(node: usize) -> bool {
    return node % 2 == 1;
}

fn left_sibling(node: usize) -> usize {
    return left_child(parent(node));
}

fn right_sibling(node: usize) -> usize {
    return right_child(parent(node));
}

fn is_leaf(node: usize, seg_array_size: usize) -> bool {
    return (node >= seg_array_size) && (node < 2 * seg_array_size);
}

fn find_left_limit(
    min_seg_tree: &SegmentTree<usize>,
    start: usize,
    alpha: usize,
) -> usize {
    let mut node = get_node_from_index(start, min_seg_tree.len);
    if min_seg_tree.tree[node] < alpha {
        return start + 1  // LCP array is shifted to the right
    }
    while (!is_root(node)) && (is_left_child(node) || (min_seg_tree.tree[left_sibling(node)] >= alpha)) {
        node = parent(node);
    }
    if is_root(node) {
        return 0usize;
    }
    node = left_sibling(node);
    while !is_leaf(node, min_seg_tree.len) {
        let rc = right_child(node);
        if min_seg_tree.tree[rc] < alpha {
            node = rc;
        }
        else {
            node = left_child(node);
        }
    }
    return get_index_from_node(node, min_seg_tree.len) + 1;
}


fn find_right_limit(
    min_seg_tree: &SegmentTree<usize>,
    start: usize,
    alpha: usize,
) -> usize {
    if start == (min_seg_tree.len - 1) {
        return min_seg_tree.len - 1;
    }
    let mut node = get_node_from_index(start + 1, min_seg_tree.len);
    if min_seg_tree.tree[node] < alpha {
        return start;
    }
    while !is_root(node) && ((is_right_child(node)) || (min_seg_tree.tree[right_sibling(node)] >= alpha)) {
        node = parent(node);
    }
    if is_root(node) {
        return min_seg_tree.len - 1;
    }
    node = right_sibling(node);
    while !is_leaf(node, min_seg_tree.len) {
        if min_seg_tree.tree[left_child(node)] < alpha {
            node = left_child(node);
        } else {
            node = right_child(node);
        }
    }
    return get_index_from_node(node, min_seg_tree.len) - 1;
}


struct StringContext {
    sa: Vec<usize>,
    sa_lookup: Vec<usize>,
    lcp: Vec<usize>,
    min_lcp_segment_tree: SegmentTree<usize>,
    partial_sa: Vec<usize>,
    min_sa_segment_tree: SegmentTree<usize>,
}

fn get_context(str: String) -> StringContext {
    let sa = generate_suffix_array_manber_myers(&str);
    let sa_lookup = inverse_permutation(&sa);
    let lcp = lcp_construction(str.as_bytes(), &sa);
    let mut _extended_size = 2;
    while _extended_size < lcp.len() {
        _extended_size *= 2;
    }
    let mut extended_lcp = vec![lcp.len(); _extended_size];
    extended_lcp[..lcp.len()].copy_from_slice(&lcp);
    let min_lcp_segment_tree = SegmentTree::from_vec(&extended_lcp, min);

    let partial_sa = vec![str.len(); str.len()];
    let min_sa_segment_tree = SegmentTree::from_vec(
        &vec![str.len(); _extended_size],
        min,
    );
    return StringContext{
        sa:sa,
        sa_lookup: sa_lookup,
        lcp: lcp,
        min_lcp_segment_tree: min_lcp_segment_tree,
        partial_sa: partial_sa,
        min_sa_segment_tree: min_sa_segment_tree,
    }
}

fn count_substrings(str: String, raw_queries: Vec<Range<usize>>) -> Vec<i64> {
    let n = str.len();
    if n == 0 {
        return vec![];
    }
    if raw_queries.len() == 0 {
        return vec![];
    }
    let StringContext {
        sa,
        sa_lookup,
        lcp: _,
        min_lcp_segment_tree,
        partial_sa: _,
        mut min_sa_segment_tree,
    } = get_context(str);

    let mut fw_counter = FWTreeWithRangedUpdates::with_len(n);

    let mut queries: Vec<Query> = Vec::with_capacity(raw_queries.len());
    for idx in 0..raw_queries.len() {
        let q = &raw_queries[idx];
        queries.push(Query{start: q.start, end: q.end, id: idx});
    }
    queries.sort_by(|a, b| a.start.cmp(&b.start));
    let mut results: Vec<Result> = Vec::with_capacity(raw_queries.len());

    let mut query = queries.pop().unwrap();
    'main_loop: for suffix in (0..n).rev() {
        let sa_pos = sa_lookup[suffix];

        let dist_elements = find_distinguished_elements(
            suffix, &min_lcp_segment_tree, &min_sa_segment_tree, &sa_lookup
        );

        fw_counter.add(suffix, suffix+1,1);

        for dist_idx in 0..dist_elements.len() {
            let _dist_sa_pos = sa_lookup[dist_elements[dist_idx]];
            let a = min(sa_pos, _dist_sa_pos) + 1;
            let b = max(sa_pos, _dist_sa_pos) + 1;
            let _progressive_lcp = min_lcp_segment_tree.query(Range{start:a, end:b}).unwrap();
            let _start = dist_elements[dist_idx] + _progressive_lcp;
            let _end =
                if dist_idx == dist_elements.len() - 1 { n }
                else { dist_elements[dist_idx + 1] + _progressive_lcp};
            fw_counter.add(_start, _end, 1);
        }

        while query.start == suffix {
            results.push(Result{value: fw_counter.range_sum(query.start, query.end), id: query.id});
            match queries.pop() {
                Some(q) => query = q,
                None => break 'main_loop,
            };
        }
        min_sa_segment_tree.update(sa_pos, suffix);
    }
    results.sort_by_key(|r| r.id);

    let mut output: Vec<i64> = Vec::with_capacity(results.len());
    for r in results {
        output.push(r.value);
    }
    return output;
}


fn parse_line_to_two_numbers() -> (i64, i64) {
    let mut line = String::new();
    io::stdin().read_line(&mut line).unwrap();
    let _tuple_sizes: Vec<&str> = line.split(" ").collect();
    let a = _tuple_sizes[0].trim().parse::<i64>().unwrap();
    let b = _tuple_sizes[1].trim().parse::<i64>().unwrap();
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
    use super::{StringContext, get_context};
    use super::min;
    use super::{find_left_limit, find_right_limit};
    use super::find_distinguished_elements;
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
    fn test_find_left_limit() {
        let lcp: Vec<usize> = vec![5, 4, 3, 2, 3, 4, 5, 5];  // Works on arrays with power-2 elements
        let min_seg_tree = SegmentTree::from_vec(&lcp, min);
        let left_1= find_left_limit(&min_seg_tree, 7, 3);
        assert_eq!(left_1, 4);
        let left_2 = find_left_limit(&min_seg_tree, 4, 3);
        assert_eq!(left_2, 4);
        let left_3 = find_left_limit(&min_seg_tree, 2, 3);
        assert_eq!(left_3, 0);
        let left_4 = find_left_limit(&min_seg_tree, 0, 0);
        assert_eq!(left_4, 0);
        let left_5 = find_left_limit(&min_seg_tree, 2, 4);
        assert_eq!(left_5, 3);
    }

    #[test]
    fn test_find_right_limit() {
        let lcp: Vec<usize> = vec![1, 3, 3, 2, 5, 2, 5, 5];  // Works only on arrays with power-2 elements
        let min_seg_tree = SegmentTree::from_vec(&lcp, min);
        let right_1= find_right_limit(&min_seg_tree, lcp.len() - 1, 4);
        assert_eq!(right_1, lcp.len() - 1);
        let right_2 = find_right_limit(&min_seg_tree, 6, 3);
        assert_eq!(right_2, 7);
        let right_3 = find_right_limit(&min_seg_tree, 4, 3);
        assert_eq!(right_3, 4);
        let right_4 = find_right_limit(&min_seg_tree, 2, 3);
        assert_eq!(right_4, 2);
        let right_5 = find_right_limit(&min_seg_tree, 1, 3);
        assert_eq!(right_5, 2);
    }

    #[test]
    fn test_find_distinguished_elements() {
        let str = String::from("abcdaababcdabc");
        let StringContext {
            sa,
            sa_lookup,
            lcp,
            min_lcp_segment_tree,
            partial_sa: _,
            mut min_sa_segment_tree,
        } = get_context(str);
        for suffix in (1..sa_lookup.len() - 1).rev() {
            min_sa_segment_tree.update(sa_lookup[suffix], suffix);
        }
        let dist_elem = find_distinguished_elements(
            0,
            &min_lcp_segment_tree,
            &min_sa_segment_tree,
            &sa_lookup,
        );
        assert_eq!(dist_elem.len(), 4);
        assert_eq!(dist_elem[0], 1);
        assert_eq!(dist_elem[1], 4);
        assert_eq!(dist_elem[2], 5);
        assert_eq!(dist_elem[3], 7);
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
            Range{start: 2, end: 4},
            Range{start: 0, end: 3},
        ];
        let expected = vec![1, 8, 1, 8, 3, 5];
        let results = count_substrings(str, queries);

        for idx in 0..results.len() {
            assert_eq!(results[idx], expected[idx]);
        }
    }

    #[test]
    fn example3() {
        let str = String::from(
            "ccccccccccgdgddgdgddgddgdgddgdgddgddgdgddgddgdgddgdgddgddgdgddwbwbbwbwbbwbbwbwbbwbkgkggwyomdjdbevunm"
        );
        let queries = vec![
            Range{start: 62, end: 70},
            Range{start: 0, end: 6},
            Range{start: 0, end: 10},
            Range{start: 8, end: 12},
        ];

        let expected = vec![24, 6, 10, 9];
        let results = count_substrings(str, queries);

        for idx in 0..results.len() {
            assert_eq!(results[idx], expected[idx]);
        }
    }

    #[test]
    fn example_tmp() {
        let str = String::from(
            "wbbwbwbbwbbw"
        );
        let queries = vec![
            Range{start: 0, end: 8},
        ];

        let expected = vec![24];
        let results = count_substrings(str, queries);

        for idx in 0..results.len() {
            assert_eq!(results[idx], expected[idx]);
        }
    }
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
