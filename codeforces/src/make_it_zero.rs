/// Solution to codeforces problem 2124E https://codeforces.com/problemset/problem/2124/E'''
use std::io::{stdin, Read, BufRead};
use std::ops::{Add, Sub};


struct TestCase {
    size: usize,
    array: Vec<i64>,
}

struct SolvedCase {
    array: Vec<i64>,
    num_steps: i32,
    step_arrays: Vec<Vec<i64>>,
}


fn read_test_case<T: Read + BufRead>(input_stream: &mut T) -> TestCase {
    let mut line = String::new();
    input_stream.read_line(&mut line).unwrap();
    let size = line.trim().parse::<usize>().unwrap();
    line = String::new();
    input_stream.read_line(&mut line).unwrap();
    let array = line.trim().split_whitespace()
        .map(|x| x.parse::<i64>().unwrap()).collect::<Vec<i64>>();
    assert_eq!(array.len(), size);
    return TestCase { size, array };
}


fn read_input<T: Read + BufRead>(input_stream: &mut T) -> Vec<TestCase> {
    let mut line = String::new();
    input_stream.read_line(&mut line).unwrap();
    let num_lines = line.trim().parse::<usize>().unwrap();
    let mut test_cases = Vec::new();
    for _ in 0..num_lines {
        test_cases.push(read_test_case(input_stream));
    }
    return test_cases;
}


fn add<T: Add<Output = T> + Copy>(a: Vec<T>, b: Vec<T>) -> Vec<T> {
    assert_eq!(a.len(), b.len());
    return a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect();
}


fn sub<T: Sub<Output = T> + Copy>(a: Vec<T>, b: Vec<T>) -> Vec<T> {
    assert_eq!(a.len(), b.len());
    return a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect();
}


fn solve<T: Read + BufRead>(input_stream: &mut T) -> Vec<SolvedCase> {
    let test_cases = read_input(input_stream);
    return Vec::new();
}


fn main() {
    solve(&mut stdin().lock());
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn example1() {
        let mut input_stream = "3\n3\n1 2 3\n2\n2 5\n4\n5 3 1 5\n".as_bytes();
        let solution = solve(&mut input_stream);
        assert_eq!(solution.len(), 3);

        assert_eq!(solution[0].num_steps, 1);
        assert_eq!(solution[0].step_arrays[0], vec![1, 2, 3]);

        assert_eq!(solution[1].num_steps, -1);

        assert_eq!(solution[2].num_steps, 2);

        let mut result = vec![0; 4];
        for step_array in solution[2].step_arrays.clone() {
            result = add(result, step_array);
        }
        assert_eq!(result, vec![5, 3, 1, 5]);
    }
}