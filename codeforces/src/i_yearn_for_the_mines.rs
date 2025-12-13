// Solution to the codeforces problem https://codeforces.com/problemset/problem/2133/E

use std::io::{stdin, Read, BufRead};
use std::collections::BTreeSet;


#[derive(Clone)]
struct Node {
    neighbours: BTreeSet<usize>,
}


#[derive(Clone)]
struct Graph {
    nodes: Vec<Node>,
}


impl Graph {
    pub fn create_disconnected_graph(n: usize) -> Self {
        let mut nodes = Vec::with_capacity(n);
        for _ in 0..n {
            nodes.push(Node { neighbours: BTreeSet::new() });
        }
        return Graph { nodes };
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.nodes[u].neighbours.insert(v);
        self.nodes[v].neighbours.insert(u);
    }

    pub fn neighbours(&self, u: usize) -> &BTreeSet<usize> {
        return &self.nodes[u].neighbours;
    }

    pub fn leaves(&self) -> Vec<usize> {
        let mut leaves = Vec::new();
        for (i, node) in self.nodes.iter().enumerate() {
            if node.neighbours.len() <= 1 {
                leaves.push(i);
            }
        }
        return leaves;
    }

    pub fn destroy_edges_connected_to_node(&mut self, u: usize) {
        for v in self.nodes[u].neighbours.clone() {
            self.nodes[v].neighbours.remove(&u);
        }
        self.nodes[u].neighbours.clear();
    }

    pub fn dfs_tour(&self, start: usize) -> Vec<usize> {
        let mut parent_queue = vec![self.nodes.len()];  // Need an additional node not in the graph
        let mut stack = vec![start];
        let mut tour = vec![];
        while stack.len() > 0 {
            let u = stack.pop().unwrap();
            tour.push(u);
            if *parent_queue.last().unwrap() == u {
                parent_queue.pop();
            }
            else {
                stack.push(u);
                for &v in self.neighbours(u) {
                    if v != *parent_queue.last().unwrap() {
                        stack.push(v);
                    }
                }
                parent_queue.push(u);
            }
            dbg!(&tour);
        }
        return tour;
    }
}


struct SolvedCase {
    steps: Vec<Action>,
}


enum ActionType {
    INSPECT,
    DESTROY,
}


struct Action {
    action_type: ActionType,
    node_idx: usize,
}

impl Action {
    pub fn to_string(&self) -> String {
        match self.action_type {
            ActionType::INSPECT => format!("1 {}", self.node_idx + 1),
            ActionType::DESTROY => format!("2 {}", self.node_idx + 1),
        }
    }
}


fn read_input<T: Read + BufRead>(input_stream: &mut T) -> Vec<Graph> {
    let mut line = String::new();
    input_stream.read_line(&mut line).unwrap();
    let num_test_cases = line.trim().parse::<usize>().unwrap();
    let mut test_cases: Vec<Graph> = Vec::with_capacity(num_test_cases);
    for _test_idx in 0..num_test_cases {
        line = String::new();
        input_stream.read_line(&mut line).unwrap();
        let num_nodes = line.trim().parse::<usize>().unwrap();
        let mut graph = Graph::create_disconnected_graph(num_nodes);
        for _graph_idx in 0..num_nodes-1 {
            line = String::new();
            input_stream.read_line(&mut line).unwrap();
            let parts = line.trim().split_whitespace()
                .map(|x| x.parse::<usize>().unwrap()).collect::<Vec<usize>>();
            assert_eq!(parts.len(), 2);
            let a = parts[0] - 1;
            let b = parts[1] - 1;
            graph.add_edge(a, b);
        }
        test_cases.push(graph);
    }
    return test_cases;
}


fn solve_test_case(graph: &mut Graph) -> SolvedCase {
    let mut actions_to_take = Vec::new();
    let start = graph.leaves()[0];
    let tour = graph.dfs_tour(start);
    let mut stack = Vec::new();
    for node in tour {
        if stack.len() < 2 {
            break;
        }
        stack.pop();
        let parent = *stack.last().unwrap();
        if graph.neighbours(node).len() == 3 {
            actions_to_take.push(Action { action_type: ActionType::DESTROY, node_idx: parent });
            graph.destroy_edges_connected_to_node(parent);

        } else if graph.neighbours(node).len() > 1 {
            actions_to_take.push(Action { action_type: ActionType::DESTROY, node_idx: node });
            graph.destroy_edges_connected_to_node(parent);
        }
    }

    let mut investigated = BTreeSet::new();
    for l in graph.leaves() {
        if investigated.contains(&l) {
            continue;
        }
        actions_to_take.push(Action { action_type: ActionType::INSPECT, node_idx: l });
        if graph.neighbours(l).len() == 0 {
            continue;
        }
        let mut previous = l;
        let mut current = *graph.neighbours(l).iter().next().unwrap();
        loop {
            actions_to_take.push(Action { action_type: ActionType::INSPECT, node_idx: current });
            investigated.insert(current);
            if graph.neighbours(current).len() == 1 {
                break;
            }
            let next = *graph.neighbours(current).iter().find(|&&x| x != previous).unwrap();
            previous = current;
            current = next;
        }
    }

    return SolvedCase { steps: actions_to_take };
}


fn main() {
    let mut test_cases = read_input(&mut stdin().lock());
    for tc in test_cases.iter_mut() {
        let solution = solve_test_case(tc);
        println!("{}", solution.steps.len());
        for step in solution.steps {
            println!("{}", step.to_string());
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dfs_tour() {
        let mut graph = Graph::create_disconnected_graph(9);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 3);
        graph.add_edge(1, 4);
        graph.add_edge(3, 5);
        graph.add_edge(3, 6);
        graph.add_edge(2, 7);
        graph.add_edge(2, 8);
        let tour = graph.dfs_tour(0);
        assert_eq!(tour, vec![0, 2, 8, 8, 7, 7, 2, 1, 4, 4, 3, 6, 6, 5, 5, 3, 1, 0]);
    }

    #[test]
    fn test_dfs_tour_2() {
        let mut graph = Graph::create_disconnected_graph(10);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge (0, 3);
        let tour = graph.dfs_tour(1);
        assert_eq!(tour, vec![1, 0, 3, 3, 2, 2, 0, 1]);
    }

    fn _assert_valid_solution(graph: &Graph, solution: &SolvedCase) {
        assert!(solution.steps.len() <= 5 * graph.nodes.len() / 4);
        let mut first_inspect_idx = 0;
        for action in &solution.steps {
            match action.action_type {
                ActionType::INSPECT => {
                    break;
                }
                ActionType::DESTROY => {
                    graph.destroy_edges_connected_to_node(action.node_idx);
                first_inspect_idx += 1;
                }
            }
        }
        for idx in 0..graph.nodes.len() {
            assert!(graph.neighbours(idx).len() <= 2);
        }
        let mut investigated = BTreeSet::new();
        for action_idx in first_inspect_idx..solution.steps.len() {
            assert!(matches!(solution.steps[action_idx].action_type, ActionType::INSPECT));
            assert!(graph.neighbours(solution.steps[action_idx].node_idx).len() <= 2);
            investigated.insert(solution.steps[action_idx].node_idx);
        }
        for idx in 0..graph.nodes.len() {
            assert!(investigated.contains(&idx));
        }
    }

    #[test]
    fn test_1() {
        let mut graph = Graph::create_disconnected_graph(4);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);

        let solution = solve_test_case(&mut graph.clone());
        _assert_valid_solution(&graph, &solution);
    }

    #[test]
    fn test_2() {
        let mut graph = Graph::create_disconnected_graph(7);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);
        graph.add_edge(1, 4);
        graph.add_edge(1, 5);
        graph.add_edge(1, 6);

        let solution = solve_test_case(&mut graph.clone());
        _assert_valid_solution(&graph, &solution);
    }

    #[test]
    fn test_3() {
        let mut graph = Graph::create_disconnected_graph(7);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);
        graph.add_edge(1, 4);
        graph.add_edge(4, 5);
        graph.add_edge(4, 6);

        let solution = solve_test_case(&mut graph.clone());
        _assert_valid_solution(&graph, &solution);
    }

}
