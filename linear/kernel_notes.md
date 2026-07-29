* First of all, I don't know how to build an MLP layer yet, so I'm trying to avoid looking at direct examples. I want to discover how it works on my own.
* First, I need to isolate the individual tensor components in PyTorch, and then hopefully, I can translate that logic into Triton.
* Why do they initialize the weight tensor in a way that requires it to be transposed?
* Ah, never mind, I already figured it out (with the help of AI). It's because the operations require contiguous memory.
* Next, I'm going to look at the matrix multiplication example in the Triton documentation; I'm not quite sharp enough to figure that part out entirely on my own.
* Actually, that example didn't help me much because they are doing a standard matrix multiplication, whereas I am dealing with batched matrix-vector multiplication.
* I'm going to check my notebook, maybe sketching it out visually will help.
* Turns out I don't have a pencil nearby, so I need to go find one first.
* I think I figured out what the grid size should be: `batch_size * (out_dim // block_size_out) * (in_dim // block_size_in)`.
* I'm taking a break now after staring at the screen and my notes for two hours straight.
* I've been chilling for a bit, so I guess the most critical step now is extracting the `batch_index`, `out_block_idx` (the row block), and `in_block_idx` (the column block). That should be doable.
* Actually, looking at it again, I feel like this grid size calculation is wrong.
* Okay, it doesn't matter turns out it *is* correct.
* Wait, I just realized I am literally just doing a regular matrix multiplication. Yeah, it took me that long to see it, and everything I've written up to this point is complete nonsense.
* The final result feels so close, yet so distant at the same time. I honestly could just copy the solution straight from the Triton docs now, but for some reason, I want to figure it out on my own.
* Okay, new day, new me. Maybe today I will be able to figure out the math behind the indexing. I will start with my notebook and try to transfer this logic into Triton.
* Okay, at least I'm doing this for a purpose (trying to understand this indexing). Since I believe they use regular GEMM, I will be doing stuff that is better suited for linear layers.
* Okay, I think I've almost got it. I'm going to go outside to get some fresh air and think about the solution, which now has a sort of latent representation in my head that I need to translate into code/words.
* I don't really understand how it is supposed to know how to calculate the dot product between A and B, or how it knows what the shapes are. Maybe this looks stupid, but I'm curious. I will use AI to ask some questions about how this works, or just try it myself and see. Actually, I won't use AI let it be a challenge for me.
* Okay, I'm in full frustration mode. Making this block indexing work is really hard. I'll try to read the tutorial code.
* Okay, I still understand nothing. I had a slight "aha" moment, but I don't know what a "group" is in this context.
* Okay, never mind. I should just read the full tutorial, not just the code. Sometimes I refuse to use resources like tutorials because I feel this impostor syndrome, like "I haven't done it myself."
* I have no idea what groups are, but a more accurate way to describe my feelings about them is: "WTF is that?"
* Okay, apparently groups are not actual groups of data; instead, they dictate the order of the data blocks.
* Okay, I decided to start over again, just doing basic tiled matrix multiplication in PyTorch.
* Okay, I finished it in an hour without many errors. I just asked AI what I did wrong, and the error turned out to be in the validation code.
* So now my main enemy is that small part of the code at the top that defines `pid_m` and `pid_n`. I'll try to copy it and adapt it to my naming conventions without changing any logic.
* Okay, this actually works! So maybe I should try printing what `pid_m` and `pid_n` actually look like.
* Aha! So it basically oscillates rows from 0 to `group_size`, multiplies them by the same column, proceeds to the next column, and when it finishes all columns, it switches over to the next set of rows.
* But I doubt whether it's actually more efficient because it looks like it's doing more work. I'll try drawing squares on a whiteboard to see how it works.
* Okay, it really loads fewer blocks and outputs the same number of blocks in the result matrix.
* Yeah, looks like I've understood it. So on to the Triton function!
* Okay, that wasn't too bad. I just moved my logic to Triton. Also, I realized I had an inefficiency in my PyTorch code because I was generating offsets inside a loop while moving horizontally. It turns out I can just create the offsets once outside the loop and add `BLOCK_SIZE_K` inside it, and it works perfectly. Yeah, I really didn't see that before.
* I'll try experimenting with different data types: what about `bf16`, `int8`, or others like `tf32`, `fp32`, and `fp16`?
* Okay, it works! Not without the help of AI, but still!
