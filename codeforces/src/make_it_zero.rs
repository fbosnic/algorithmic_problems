/// Solution to codeforces problem 2124E https://codeforces.com/problemset/problem/2124/E'''
use std::io::{stdin, Read, BufRead};
use std::ops::{Add, Sub};


struct TestCase {
    size: usize,
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
        if 2 * (mass_to_the_left + array[idx]) > mass {
            return idx;
        }
        mass_to_the_left += array[idx];
    }
    return array.len();
}

fn solve_test_case(test_case: TestCase) -> SolvedCase {
    let mut array = test_case.array.clone();
    let mut steps_array:Vec<Vec<i64>> = vec![];

    let mut total_mass: i64 = compute_mass(&array, 0, array.len());
    let max_element: i64 = _max_element(&array);
    if (total_mass % 2 == 1) || (total_mass < 2 * max_element) {
        return SolvedCase { num_steps: -1, step_arrays: Vec::new() };
    }

    while total_mass > 0 {
        total_mass = compute_mass(&array, 0, array.len());
        let left_half_idx = _find_half_mass(&array, 0, total_mass);
        let left_mass = compute_mass(&array, 0, left_half_idx);
        if 2 * left_mass == total_mass {
            let s_array = array.clone();
            steps_array.push(s_array);
            break;
        }
        let mut s_array = vec![0; array.len()];
        for idx in 0..left_half_idx {
            s_array[idx] = array[idx];
        }
        let remaining_mass = total_mass - left_mass;
        let next_mid_idx = _find_half_mass(&array, left_half_idx, remaining_mass);
        let middle_mass = compute_mass(&array, left_half_idx, next_mid_idx);
        if 2 * middle_mass < remaining_mass - left_mass {
            s_array[next_mid_idx] = left_mass;
        }
        else {
            s_array[next_mid_idx] = remaining_mass - 2 * middle_mass;
            let to_distribute = left_mass - s_array[next_mid_idx];
            assert_eq!(to_distribute % 2, 0);
            let mut _to_distribut_left = to_distribute / 2;
            let mut _to_distribute_right = to_distribute / 2;

            for idx in left_half_idx..next_mid_idx {
                if array[idx] < _to_distribut_left {
                    _to_distribut_left -= array[idx];
                    s_array[idx] = array[idx];
                }
                else {
                    s_array[idx] = _to_distribut_left;
                    _to_distribut_left = 0;
                }
            }

            for idx in next_mid_idx..array.len() {
                let mut _updated_array_value = array[idx] - s_array[idx];
                if _updated_array_value < _to_distribute_right {
                    _to_distribute_right -= _updated_array_value;
                    s_array[idx] += _updated_array_value;
                }
                else {
                    s_array[idx] += _to_distribute_right;
                    _to_distribute_right = 0;
                }
            }
        }
        array = sub(&array, &s_array);
        steps_array.push(s_array);
    }

    return SolvedCase { num_steps: steps_array.len() as i32, step_arrays: steps_array };
}


fn solve<T: Read + BufRead>(input_stream: &mut T) -> Vec<SolvedCase> {
    let test_cases = read_input(input_stream);
    let mut results: Vec<SolvedCase> = vec![];
    for tc in test_cases {
        results.push(solve_test_case(tc));
    }
    return results;
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