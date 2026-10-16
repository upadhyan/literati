pub mod and_state;
pub mod arena;
pub mod data_state;
pub mod shape_fn_state;

/// Flat representation of per-bin sample indices.
/// Replaces Vec<Vec<u32>> with a single contiguous Vec + offset array.
#[derive(Clone, Debug)]
pub struct FlatBins {
    pub data: Vec<u32>,
    pub offsets: Vec<u32>,
}

impl FlatBins {
    pub fn bin(&self, i: usize) -> &[u32] {
        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;
        &self.data[start..end]
    }

    pub fn n_bins(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn from_bins(bins: &[Vec<u32>]) -> Self {
        let total_len: usize = bins.iter().map(|b| b.len()).sum();
        let mut offsets = Vec::with_capacity(bins.len() + 1);
        let mut data = Vec::with_capacity(total_len);
        offsets.push(0u32);
        for bin in bins {
            data.extend_from_slice(bin);
            offsets.push(data.len() as u32);
        }
        FlatBins { data, offsets }
    }

    pub fn data_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<u32>()
    }
}

/// Typed indices into arena Vecs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DataStateId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ShapeFnStateId(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AndStateId(pub u32);

/// Reference to a child in a DataState's queue (shape function children).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DataChildRef {
    ShapeFn(ShapeFnStateId),
}

/// Reference to a child in a ShapeFunctionState's queue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ShapeFnChildRef {
    And(AndStateId),         // first-order: the actual split
    ShapeFn(ShapeFnStateId), // higher-order refinement
}

/// Parent reference — used for backpropagation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParentRef {
    None,
    DataState(DataStateId),
    ShapeFnState(ShapeFnStateId),
}
