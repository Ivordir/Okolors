const workgroup_size = 64u;
const max_k = 256u;

// Future: make these override variables once wgpu supports them
struct Globals {
    k: u32,
    n: u32,
}

struct State {
    curr_k: u32,
    n_remainder: u32,
    curr_random: u32,
}

struct OklabN {
    lab: vec3f,
    n: u32,
}

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var<storage, read> oklab: array<OklabN>;
@group(0) @binding(2) var<uniform> random_centroids: array<OklabN, max_k>;
@group(0) @binding(3) var<storage, read_write> centroids: array<OklabN>;
@group(0) @binding(4) var<storage, read_write> sums: array<OklabN>;
@group(0) @binding(5) var<storage, read_write> assignment: array<u32>;
@group(0) @binding(6) var<storage, read_write> state: State;

fn squared_distance(x: vec3f, y: vec3f) -> f32 {
    let diff = x - y;
    return dot(diff, diff);
}

@compute @workgroup_size(64)
fn lloyd_update_assignments(@builtin(global_invocation_id) id: vec3u) {
    if id.x >= globals.n {
        return;
    }

    let k = globals.k;
    let color = oklab[id.x].lab;

    var min_dist = squared_distance(color, centroids[0].lab);
    var min_center = 0u;
    for (var i = 1u; i < k; i++) {
        let dist = squared_distance(color, centroids[i].lab);
        if dist < min_dist {
            min_dist = dist;
            min_center = i;
        }
    }

    assignment[id.x] = min_center;
}

fn add_oklabn(sum: OklabN, x: OklabN) -> OklabN {
    return OklabN(sum.lab + x.lab, sum.n + x.n);
}

fn scale_up(x: OklabN) -> OklabN {
    return OklabN(x.lab * vec3(f32(x.n)), x.n);
}

fn scale_down(x: OklabN) -> OklabN {
    return OklabN(x.lab / vec3(f32(x.n)), x.n);
}

// The code below implement some of the techniques from:
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
// Though, it remains to be seen whether this still applies to more modern hardware and drivers:
// https://diaryofagraphicsprogrammer.blogspot.com/2014/03/compute-shader-optimizations-for-amd.html

var<workgroup> partial_sum: array<OklabN, workgroup_size>;

@compute @workgroup_size(64)
fn lloyd_update_centroids_first_pass(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
) {
    let k = state.curr_k;
    let id = local_id.x;

    if (workgroup_id.x != num_workgroups.x - 1u || id < state.n_remainder) && k == assignment[global_id.x] {
        partial_sum[id] = scale_up(oklab[global_id.x]);
    }
    // else, partial_sum[id] remains zero value
    workgroupBarrier();

    for (var i = workgroup_size / 2u; i > 0u; i >>= 1u) {
        if id < i {
            partial_sum[id] = add_oklabn(partial_sum[id], partial_sum[id + i]);
        }
        workgroupBarrier();
    }

    if id == 0u {
        sums[workgroup_id.x] = partial_sum[0];

        if workgroup_id.x == num_workgroups.x - 1u {
            state.n_remainder = num_workgroups.x % workgroup_size;
        }
    }
}

@compute @workgroup_size(64)
fn lloyd_update_centroids(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(local_invocation_id) local_id: vec3u,
    @builtin(workgroup_id) workgroup_id: vec3u,
    @builtin(num_workgroups) num_workgroups: vec3u,
) {
    let id = local_id.x;

    if workgroup_id.x != num_workgroups.x - 1u || id < state.n_remainder {
        partial_sum[id] = sums[global_id.x];
    }
    // else, partial_sum[id] remains zero value
    workgroupBarrier();

    for (var i = workgroup_size / 2u; i > 0u; i >>= 1u) {
        if id < i {
            partial_sum[id] = add_oklabn(partial_sum[id], partial_sum[id + i]);
        }
        workgroupBarrier();
    }

    if id == 0u {
        if num_workgroups.x == 1u {
            let sum = partial_sum[0];
            if sum.n == 0u {
                centroids[state.curr_k] = random_centroids[state.curr_random];
                state.curr_random++;
            } else {
                centroids[state.curr_k] = scale_down(sum);
            }

            if state.curr_k == globals.k - 1u {
                state.curr_k = 0u;
            } else {
                state.curr_k++;
            }

            state.n_remainder = globals.n % workgroup_size;
        } else {
            sums[workgroup_id.x] = partial_sum[0];

            if workgroup_id.x == num_workgroups.x - 1u {
                state.n_remainder = num_workgroups.x % workgroup_size;
            }
        }
    }
}
