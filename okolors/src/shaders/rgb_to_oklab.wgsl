@group(0) @binding(0) var<storage, read> srgb: array<vec3f>;
@group(0) @binding(1) var<storage, read_write> oklab: array<vec3f>;

fn linear(x: f32) -> f32 {
    if x >= 0.04045 {
        // return pow((x + 0.055) / 1.055, 2.4);
        return pow(fma(x, 1.0 / 1.055, 0.055 / 1.055), 2.4);
    } else {
        return x / 12.92;
    }
}

fn cbrt(x: f32) -> f32 {
    return pow(x, 1.0 / 3.0);
}

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let srgb = srgb[id.x];
    let m1 = mat3x3f(
        vec3(0.4122214708, 0.2119034982, 0.0883024619),
        vec3(0.5363325363, 0.6806995451, 0.2817188376),
        vec3(0.0514459929, 0.1073969566, 0.6299787005)
    );
    let m2 = mat3x3f(
        vec3(0.2104542553, 1.9779984951, 0.0259040371),
        vec3(0.7936177850, -2.4285922050, 0.7827717662),
        vec3(-0.0040720468, 0.4505937099, -0.8086757660)
    );

    let linear = vec3(linear(srgb.r), linear(srgb.g), linear(srgb.b));
    let lms = m1 * linear;
    let lms_ = vec3(cbrt(lms.x), cbrt(lms.y), cbrt(lms.z));
    oklab[id.x] = m2 * lms_;
}
