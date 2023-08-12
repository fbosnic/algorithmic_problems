use std::io::{self, BufRead, Write};
use std::vec;
use std::convert::TryFrom;

fn compute_polylin_fn(fn_args: &Vec<i64>, fn_vals: &Vec<i64>, x: i64) -> i64 {
    let min = *fn_args.first().unwrap();
    let max = *fn_args.last().unwrap();
    let num_el = i64::try_from(fn_args.len()).unwrap();
    if x <= min {
        return *fn_vals.first().unwrap() + num_el * (min - x);
    } else if x >= max {
        return *fn_vals.last().unwrap() + num_el * (x - max);
    } else {
        let i = fn_args.binary_search(&x);
        let i = match i {
            Err(e) => e,
            Ok(v) => v,
        };
        let space = fn_args[i] - fn_args[i - 1];
        if space == 0 {
            return fn_vals[i];
        }
        let slope = i64::try_from((fn_vals[i] - fn_vals[i - 1]) / space).unwrap();
        return fn_vals[i - 1] + slope * (x - fn_args[i - 1]);
    }
}

fn playing_with_numbers(mut arr: Vec<i64>, queries: Vec<i64>) -> Vec<i64> {
    arr.sort_unstable();
    let arr_len = arr.len();
    let mut polylin = vec![0; arr_len];
    for i in 1..arr_len {
        polylin[i] = polylin[i - 1] + (arr[i] - arr[i - 1]) * i64::try_from(i).unwrap();
    }
    let mut _cumulative = 0;
    for i in (0..(arr_len - 1)).rev() {
        _cumulative += (arr[i + 1] - arr[i]) * i64::try_from(arr_len - 1 - i).unwrap();
        polylin[i] += _cumulative;
    }

    let mut x = 0;
    let mut res = Vec::with_capacity(arr.len());
    for q in queries {
        x -= q;
        res.push(compute_polylin_fn(&arr, &polylin, x));
    }
    res
}

fn main() {
    let stdin = io::stdin();
    let mut stdin_iterator = stdin.lock().lines();

    // let mut fptr = File::create(env::var("OUTPUT_PATH").unwrap()).unwrap();
    let mut fptr = std::io::stdout();

    let _ = stdin_iterator
        .next()
        .unwrap()
        .unwrap()
        .trim()
        .parse::<i64>()
        .unwrap();

    let arr: Vec<i64> = stdin_iterator
        .next()
        .unwrap()
        .unwrap()
        .trim_end()
        .split(' ')
        .map(|s| s.to_string().parse::<i64>().unwrap())
        .collect();

    let _ = stdin_iterator
        .next()
        .unwrap()
        .unwrap()
        .trim()
        .parse::<i64>()
        .unwrap();

    let queries: Vec<i64> = stdin_iterator
        .next()
        .unwrap()
        .unwrap()
        .trim_end()
        .split(' ')
        .map(|s| s.to_string().parse::<i64>().unwrap())
        .collect();

    let result = playing_with_numbers(arr, queries);

    for i in 0..result.len() {
        write!(&mut fptr, "{}", result[i]).ok();

        if i != result.len() - 1 {
            writeln!(&mut fptr).ok();
        }
    }

    writeln!(&mut fptr).ok();
}
