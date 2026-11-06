/// Element data type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    /// 4-bit affine quantization (grouped).
    Q4,
}

/// Tensor shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn numel(&self) -> usize {
        self.0.iter().product()
    }
}

/// A simple tensor container (storage is raw bytes).
#[derive(Debug)]
pub struct Tensor {
    pub dtype: DType,
    pub shape: Shape,
    pub data: Vec<u8>,
}

impl Tensor {
    pub fn new(dtype: DType, shape: Shape, data: Vec<u8>) -> Self {
        Self { dtype, shape, data }
    }
}
