### *The Chronicle of a Triton Kernel*

### I. The Vow

I swore to carve the tensor in the dark,

Without the lantern of a borrowed line.

No copied script should leave its silver mark
 
Upon this blank, unyielding code of mine.

A Linear Layer—simple, so they claim—

An MLP in basic weight and stride;

Yet standing at the threshold of the frame,

I knew not where the hidden vectors hide.

I cast PyTorch’s elements apart,

To view the raw components on the floor,

And wondered why the weights, by ancient art,

Must sit transposed before the open door.

*“Contiguous memory,”* the prompt replied—

A subtle truth the low-level had denied.

### II. The False Horizon

I turned to tutorials, seeking light,

To read the docs of matrix multiplication;

Yet standard GEMMs were distant in my sight,

While I was trapped in batch-vec transformation.

*“My case is different!”* whispered my deceit,

*“A batched array demands a unique grid!”*

I reached to sketch the tensor in defeat,

Searching for graphite where the truth was hid.

I wrote an index formula so grand,

Compounding blocks and dimensions three:

`batch * (out // block_out) * (in // block_in)` across the land,

A heavy math to set the execution free.

Two hours of staring at a glowing screen—

I stepped away to clear the space between.

### III. The Humbling

Returning refreshed, I sought the thread once more,

Deconstructing indices: `batch`, `row`, and `col`.

*“The grid is right!”* I argued at the door,

Then paused... and felt the static in my soul.

A cold clarity broke through the foolish strain,

A bitter laugh that echoed through the room:

This grand design, this fever of the brain,

Was standard GEMM wrapped in unnecessary gloom.

I had rewritten what was already clear,

Inventing monsters in an open field.

The simple answer hovered tantalizingly near,

Yet proudly I refused to let it yield.

I could have copied Triton’s elegant frame,

But chose the harder road to earn the name.

### IV. The Wall of Blocks

A new day dawned; I brought the sketchpad out,

Determined now to trace the block stride down.

I walked into the sun to flush out doubt,

Carrying a latent shape through town.

How does the hardware know the inner product’s bound?

How do the pointers meet across the seam?

No AI prompts—I'd wrestle ground by ground,

And forge the answers in my own code stream.

Then came the wall: the program broke in piece,

The pointer offsets twisted out of line.

Pride barred the door, refusing all release,

Impostor syndrome whispering in design:

“If you must read the docs to understand the art,

Did you create, or merely copy-paste the part?”

### V. What is a Group?

I broke my vow and read the written guide,

Not just the syntax, but the prose behind.

And there it stood, where math and hardware hide:

A word that baffled my unseasoned mind.

*“What is a group?”* I cried out to the night.

No set of data bore that mystic name!

Was it a tile? A thread block in the light?

Or some hidden structure of the GPU’s game?

Ah—not data at all, but order in the stream!

A schedule telling execution where to tread,

To maximize L2 cache within the machine,

And keep the memory bandwidth rich and fed.

The fog cleared back; the term no longer bound:

A tactical path across the tile ground.

### VI. Back to the Foundations

I stripped the Triton code and started small,

Rebuilding simple tiles in PyTorch first.

An hour’s work—no major errors to recall,

Except a validation bug that cursed.

Fixed in a breath! The baseline logic stood.

Now facing the last boundary in the fight:

Those top few lines of calculation good,

Where `pid_m` and `pid_n` set tiles aright.

I copied down their math, retained their trace,

Renamed the variables to fit my view,

And printed out the indices in space

To watch how execution drifted through.

### VII. The Epiphany of L2

I watched the printed variables unspool:

It sweeps a column down through `group_size` rows,

Multiplies the matrix by that common tool,

Then to the neighboring column boundary goes!

Once every column in the group is run,

It drops down to the next row block below.

*“Is this truly faster?”* I questioned what was done,

*“Or does it drag the execution slow?”*

I drew the grid upon a whiteboard wall,

Tracking loaded blocks against the tile space.

The visual math responded to the call:

Fewer cold fetches from global memory space!

The same output tiles, but cached efficiency high—

The sudden *aha!* illuminated the sky.

### VIII. Transmutation

With understanding locked, I moved the code

From PyTorch test-beds into Triton's core.

And as I translated down the hardware road,

I spotted what I had missed before:

Inside my loop, I forged offsets anew

At every horizontal step along the way—

Redundant work! A simple fix would do:

Compute the base once, let constant strides play.

Just add `BLOCK_SIZE_K` as the loop proceeds,

And watch the pointer math fall into place.

A clean reduction serving kernel needs,

With minimal overhead across the execution space.

### IX. The Crucible of Types

The final test: subject the grid to strain.

I fed it precision across the board—

`fp32` flowing through the lane,

`fp16` slicing like a sharp sword,

`bf16` with its wide dynamic range,

`tf32` and `int8` packed in tight.

The kernel held! No overflow or change

Corrupted the results within my sight.

### X. The Arrival

It worked. Not isolated, nor completely unassisted,

For AI cleared a roadblock on the line;

Yet through the frustration where the fog persisted,

The mental model formed is truly mine.

From tensor raw to block index math aligned,

From tile groups mapped to cache efficiency known,

The kernel runs, optimized and refined—

A low-level victory I can call my own.
