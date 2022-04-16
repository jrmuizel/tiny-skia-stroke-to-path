/*!
`tiny-skia` is a tiny [Skia](https://skia.org/) subset ported to Rust.

`tiny-skia` API is a bit unconventional.
It doesn't look like cairo, QPainter (Qt), HTML Canvas or even Skia itself.
Instead, `tiny-skia` provides a set of low-level drawing APIs
and a user should manage the world transform, clipping mask and style manually.

See the `examples/` directory for usage examples.
*/

#![no_std]
#![warn(missing_docs)]
#![warn(missing_copy_implementations)]
#![warn(missing_debug_implementations)]

#![allow(clippy::approx_constant)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::eq_op)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::float_cmp)]
#![allow(clippy::identity_op)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::too_many_arguments)]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("You have to activate either the `std` or the `libm` feature.");

#[cfg(feature = "std")]
extern crate std;

extern crate alloc;

mod dash;

mod floating_point;
mod geom;
mod path;
mod path_builder;
mod path_geometry;
mod scalar;
mod stroker;
mod transform;
mod wide;

pub use dash::{StrokeDash, dash};
use geom::{Rect, Point};
pub use path::{Path, PathSegment, PathSegmentsIter};
pub use path_builder::PathBuilder;
pub use stroker::{LineCap, LineJoin, Stroke, PathStroker};
use transform::Transform;

/// An integer length that is guarantee to be > 0
type LengthU32 = core::num::NonZeroU32;
