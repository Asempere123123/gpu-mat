@group(0) @binding(0)
var<storage, read_write> a: array<f32>;

@group(0) @binding(1)
var<storage, read> b: array<f32>;

@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let array_length = arrayLength(&a);
    if (idx >= array_length) {
        return;
    }

    a[idx] += b[idx];
}
