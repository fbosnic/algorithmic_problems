// This is the solution to the codeforces problem https://codeforces.com/problemset/problem/2111/F

use std::io::{BufRead, Read, stdin};
use std::collections::HashSet;
use std::cmp::{min, max};


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Fraction {
    num: i32,
    den: i32,
}


fn gcd(a: i32, b: i32) -> i32 {
    let mut x = max(a, b);
    let mut y = min(a, b);
    assert!(y > 0);
    while x % y != 0 {
        let rem = x % y;
        x = y;
        y = rem;
    }
    return y;
}


impl Fraction {
    fn simplify(&mut self) {
        let gcd = gcd(self.num, self.den);
        self.num /= gcd;
        self.den /= gcd;
    }
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Coordinate {
    x: i32,
    y: i32,
}


#[derive(Debug, Clone)]
struct Solution {
    num_steps: i32,
    blocks: Vec<Coordinate>,
}


fn read_input<T: Read + BufRead>(
    input_stream: &mut T
) -> Vec<Fraction> {
    let mut input = String::new();
    input_stream.read_line(&mut input).unwrap();
    let n = input.trim().parse::<usize>().unwrap();
    let mut test_cases = Vec::with_capacity(n);
    for _ in 0..n {
        input = String::new();
        input_stream.read_line(&mut input).unwrap();
        let parts = input.trim().split_whitespace()
            .map(|x| x.parse::<i32>().unwrap()).collect::<Vec<i32>>();
        assert_eq!(parts.len(), 2);
        test_cases.push(Fraction { num: parts[0], den: parts[1] });
    }
    return test_cases;
}


fn solve(frac: &mut Fraction) -> Solution {
    frac.simplify();
    if frac.num % 2 != 0 {
        frac.num *= 2;
        frac.den *= 2;
    }
    if 2 * (frac.den + 1) < frac.num {
        return Solution { num_steps: -1, blocks: Vec::new() };
    }
    let mut a = (frac.num - 2) / 4;
    let b = (frac.num - 2) - a;
    let k = frac.den / a * b + match frac.den % a {0 => 0, _ => 1};  // ceil operation
    frac.num *= k;
    frac.den *= k;
    a *= k;

    let mut blocks = Vec::new();
    for idx in 0..frac.den {
        blocks.push(Coordinate { x:idx / a , y: idx % a });
    }
    return Solution { num_steps: blocks.len().try_into().unwrap(), blocks: blocks };
}


fn main() {
    let mut test_cases = read_input(&mut stdin().lock());
    for frac in test_cases.iter_mut() {
        let solution = solve(frac);
        println!("{}", solution.num_steps);
        for block in solution.blocks {
            println!("{} {}", block.x, block.y);
        }
    }
}


fn validate_solution(frac: Fraction, solution: Solution) {
    let den: i32 = solution.blocks.len().try_into().unwrap();
    let mut num = 0;
    let mut placed: HashSet<Coordinate> = HashSet::new();
    for block in solution.blocks {
        num += 4;
        for x_dis in [-1, 1] {
            for y_dis in [-1, 1] {
                if placed.contains(&Coordinate { x: block.x + x_dis, y: block.y + y_dis }) {
                    num -= 2;
                }
            }
        }
        placed.insert(block);
        num += 1;
    }
    assert_eq!(num * frac.den, den * frac.num);
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_input() {
        let input = "2\n1 2\n3 4\n";
        let mut input_stream = input.as_bytes();
        let test_cases = read_input(&mut input_stream);
        assert_eq!(test_cases.len(), 2);
        assert_eq!(test_cases[0], Fraction { num: 1, den: 2 });
        assert_eq!(test_cases[1], Fraction { num: 3, den: 4 });
    }
    #[test]
    fn test_1() {
        let mut frac = Fraction { num: 9, den: 2 };
        let solution = solve(&mut frac);
        assert_eq!(solution.num_steps, -1);
    }

    #[test]
    fn test_2() {
        let mut frac: Fraction = Fraction { num: 23, den: 17 };
        let solution = solve(&mut frac);
        validate_solution(frac, solution);
    }

    #[test]
    fn test_3() {
        let mut frac: Fraction = Fraction { num: 14, den: 4 };
        let solution = solve(&mut frac);
        validate_solution(frac, solution);
    }

    #[test]
    fn test_4() {
        let mut frac: Fraction = Fraction { num: 8, den: 2 };
        let solution = solve(&mut frac);
        validate_solution(frac, solution);
    }
}
