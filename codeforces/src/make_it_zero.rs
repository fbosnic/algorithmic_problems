/// Solution to codeforces problem 2124E https://codeforces.com/problemset/problem/2124/E'''
use std::io::{stdin, Read, BufRead, Write, stdout};
use std::ops::{Add, Sub};


struct TestCase {
    array: Vec<i64>,
}

struct SolvedCase {
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
    return TestCase { array };
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


fn sub<T: Sub<Output = T> + Copy>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    assert_eq!(a.len(), b.len());
    return a.iter().zip(b.iter()).map(|(x, y)| *x - *y).collect();
}

fn compute_mass(array: &Vec<i64>, left_idx: usize, right_idx: usize) -> i64 {
    let mut mass = 0;
    for idx in left_idx..right_idx {
        mass += array[idx];
    }
    return mass;
}

fn _max_element(array: &Vec<i64>) -> i64 {
    return *array.iter().max().unwrap();
}

fn _find_half_mass(array: &Vec<i64>, left_idx: usize, mass: i64) -> usize {
    let mut mass_to_the_left = 0;
    for idx in left_idx..array.len() {
        mass_to_the_left += array[idx];
        if 2 * mass_to_the_left > mass {
            return idx;
        }
    }
    return array.len();
}

fn solve_test_case(test_case: TestCase) -> SolvedCase {
    let array = test_case.array.clone();

    let total_mass: i64 = compute_mass(&array, 0, array.len());
    let max_element: i64 = _max_element(&array);
    if (total_mass % 2 == 1) || (total_mass < 2 * max_element) {
        return SolvedCase { num_steps: -1, step_arrays: Vec::new() };
    }

    let mut sub_array = vec![0; array.len()];
    let middle_idx = _find_half_mass(&array, 0, total_mass);
    let left_mass = compute_mass(&array, 0, middle_idx);
    if 2 * left_mass == total_mass {
        let s_array = array.clone();
        return SolvedCase { num_steps: 1, step_arrays: vec![s_array] };
    }

    let right_mass = total_mass - left_mass - array[middle_idx];

    let mut to_redistribute = (left_mass + array[middle_idx] - right_mass) / 2;
    sub_array[middle_idx] = to_redistribute;
    for idx in 0..middle_idx {
        if array[idx] < to_redistribute {
            to_redistribute -= array[idx];
            sub_array[idx] = array[idx];
        }
        else {
            sub_array[idx] = to_redistribute;
            break;
        }
    }

    let reminder = sub(&array, &sub_array);
    return SolvedCase {
        num_steps: 2,
        step_arrays: vec![sub_array, reminder],
     };
}


fn solve<T: Read + BufRead>(input_stream: &mut T) -> Vec<SolvedCase> {
    let test_cases = read_input(input_stream);
    let mut results: Vec<SolvedCase> = vec![];
    for tc in test_cases {
        results.push(solve_test_case(tc));
    }
    return results;
}


fn print_result<T: Write>(results: Vec<SolvedCase>, output_stream: &mut T) {
    for result in results {
        write!(output_stream, "{}\n", result.num_steps).unwrap();
        if result.num_steps == -1 {
            continue;
        }
        for step_array in result.step_arrays {
            let s = step_array.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" ");
            write!(output_stream, "{s}\n").unwrap();
        }
    }
}


fn main() {
    let results = solve(&mut stdin().lock());
    print_result(results, &mut stdout().lock());
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

    #[test]
    fn test_large_input() {
        let input = vec![100, 100, 2, 2, 50, 2, 2, 20];
        let mut input_string = input.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" ");
        input_string = format!("1\n{}\n{}\n", input.len(), input_string);
        let mut input_stream = input_string.as_bytes();
        let solution = solve(&mut input_stream);
        let mut result = vec![0; 8];
        for sa in solution[0].step_arrays.iter() {
            dbg!(sa);
            result = add(result, sa.clone());
        }
        assert_eq!(result, input);
    }


    #[test]
    fn test_output() {
        let mut input_stream = "3\n3\n1 2 3\n2\n2 5\n6\n5 5 2 2 1 1".as_bytes();
        let solution = solve(&mut input_stream);
        let mut output_stream = Vec::new();
        print_result(solution, &mut output_stream);
        let output_string = String::from_utf8(output_stream).unwrap();
        assert_eq!(output_string, "1\n1 2 3\n-1\n2\n2 2 0 0 0 0\n3 3 2 2 1 1\n");
    }
}
