use std::cmp::min;
use std::vec::Vec;
use std::ops::{Range, Add, AddAssign, Neg, Mul};

extern crate algorithms;
use algorithms::data_structures::{RBTree, FenwickTree};


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
    for i in 1..n {
        inverse_permutation[permutation[i]] = i;
    }
    return inverse_permutation;
}


struct AdvanceFenwickTree<T: Add + AddAssign + Copy + Default + Neg> {
    linear_fwt: FenwickTree<T>,
    const_fwt: FenwickTree<T>
}


impl<T: Add<Output = T> + AddAssign + Copy + Default + Neg<Output = T> + Mul<i32, Output = T>> AdvanceFenwickTree<T> {
    pub fn add(&mut self, start: usize, end: usize, value: T) {
        if start >= end {
            return;
        }
        self.linear_fwt.add(start, value);
        self.linear_fwt.add(end, -value);
        self.const_fwt.add(start, -value * (i32::try_from(start).unwrap() - 1));
        self.const_fwt.add(end, value * (i32::try_from(end).unwrap() - 1));
    }

    pub fn prefix_sum(&self, end: usize) -> T {
        let lin_part = self.linear_fwt.prefix_sum(end) * i32::try_from(end).unwrap();
        let const_part = self.const_fwt.prefix_sum(end);
        return lin_part + const_part;
    }

    pub fn range_sum(&self, start: usize, end:usize) -> T {
        return self.prefix_sum(end) + (- self.prefix_sum(start));
    }

    pub fn with_len(len: usize) -> Self {
        AdvanceFenwickTree {
            linear_fwt: FenwickTree::with_len(len),
            const_fwt: FenwickTree::with_len(len)
        }
    }
}

fn _update_prefix_counter_with(fw_counter: &mut AdvanceFenwickTree<i32>, left_len: usize, right_len: usize, comon_prefix: usize, value: i32) {
    let min_len = min(left_len, right_len);
    fw_counter.add(0, 1, value * i32::try_from(comon_prefix).unwrap());
    fw_counter.add(min_len - comon_prefix + 1, min_len + 1, -value);
}

fn add_prefix(fw_counter: &mut AdvanceFenwickTree<i32>, left_len: usize, right_len: usize, comon_prefix: usize) {
    _update_prefix_counter_with(fw_counter, left_len, right_len, comon_prefix, 1);
}

fn remove_prefix(fw_counter: &mut AdvanceFenwickTree<i32>, left_len: usize, right_len: usize, comon_prefix: usize) {
    _update_prefix_counter_with(fw_counter, left_len, right_len, comon_prefix, -1);
}


struct Query {
    start: usize,
    end: usize,
    id: usize,
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

    let sa = algorithms::string::generate_suffix_array_manber_myers(&str);
    let sa_lookup = inverse_permutation(&sa);
    let lcp = lcp_construction(str.as_bytes(), &sa);
    let min_lcp_segment_tree = algorithms::data_structures::SegmentTree::from_vec(&lcp, min);

    let mut suffix_sorter: RBTree<usize, i8> = RBTree::new();
    let mut fw_prefix_counter: AdvanceFenwickTree<i32> = AdvanceFenwickTree::with_len(n);

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
        suffix_sorter.insert(sa_pos, 0);
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
            add_prefix(
                &mut fw_prefix_counter,
                n - sa[sa_pos],
                n - sa[sa_right],
                cp
            );
        }

        while query.start == suffix {
            let max_substrings = i32::try_from((query.end - query.start) * (query.end - query.start + 1) / 2).unwrap();
            let res = max_substrings - fw_prefix_counter.prefix_sum(n - query.end);
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


fn main() {
    let str = String::from("aaabbabaaa");
    let queries = vec![
        Range{start: 8, end: 10},
        Range{start: 7, end: 9},
        Range{start: 5, end: 9},
    ];
    let results = count_substrings(str, queries);

    for r in results {
        println!("{r}");
    }
}


#[cfg(test)]
mod tests {
    use super::count_substrings;
    use std::ops::Range;

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
}
