enable f16;

@group(0) @binding(0)
var<storage, read> a: array<f16>;
@group(0) @binding(1)
var<storage, read> b: array<f16>;

@group(0) @binding(2)
var<storage, read_write> output: array<f16>;

@compute @workgroup_size(64)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    let array_length = arrayLength(&output);
    if (idx >= array_length) {
        return;
    }

    output[idx] = a[idx] + b[idx];
}
