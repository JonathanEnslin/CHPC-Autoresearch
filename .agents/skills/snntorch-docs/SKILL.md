---
name: snntorch-docs
description: Search the repository's bundled snntorch RST documentation and explain its neuron classes, API, equations, and examples. Use when asked about snntorch primitives such as Leaky, RLeaky, SLSTM, surrogate, or spikegen.
---

Search `docs/snntorch/docs/` for the requested topic. If no topic is provided, list the available RST documentation files.

Prefer files whose names match the requested class. Present a concise explanation retaining useful parameter descriptions, equations, and examples while removing unhelpful RST directive boilerplate. Cite the local source path. If no match exists, say so and offer related terms from the file listing.
