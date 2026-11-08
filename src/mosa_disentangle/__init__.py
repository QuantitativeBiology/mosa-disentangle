"""Disentangled latent blocks for the MOSA multi-omics VAE.

The package extends the public ``mosalib`` VAE with a latent space organised
into factor-specific blocks (technical source, tissue, EMT, ...) plus a free
block, and with the regularisers needed to keep each factor inside its block:
supervised heads, leakage discriminators and HSIC/CKA independence penalties.
"""

__version__ = "0.1.0"
