use std::io::{self, BufRead};

/*
 * Complete the 'stringSimilarity' function below.
 *
 * The function is expected to return an INTEGER.
 * The function accepts STRING s as parameter.
 */

fn string_similarity(s: &str) -> usize {
    let z_fn = compute_z_fn(s);
    return z_fn.into_iter().sum();
}

fn compute_z_fn(s: &str) -> Vec<usize> {
    let s: Vec<char> = s.chars().collect();
    let len = s.len();
    let mut z_fn = vec![0; len];
    let mut l: usize = 0;
    let mut r: usize = 1;
    for i in 1..len {
        if z_fn[i-l] < r-i {
            z_fn[i] = z_fn[i-l];
        } else {
            while (r < len) && (s[r] == s[r-i]) {
                r += 1;
                l = i;
            }
            z_fn[i] = r - i;
            if i == r {
                l = i;
                r = i + 1;
            }
        }
    }
    z_fn[0] = len;
    return z_fn;

}

fn main() {
    let stdin = io::stdin();
    let mut stdin_iterator = stdin.lock().lines();
    let t = stdin_iterator.next().unwrap().unwrap().trim().parse::<i32>().unwrap();
    for _ in 0..t {
        let s = stdin_iterator.next().unwrap().unwrap();
        let result = string_similarity(&s);
        println!("{result}");
    }
}

