# Migration from 0.1 to 0.2

Version 0.2 exposes the installable package as `adaptive_surrogate`.

```python
# 0.1 repository-checkout import
from src.core_engine import AdaptiveBlackBox

# 0.2 installed-package import
from adaptive_surrogate import AdaptiveBlackBox
```

The checkout-level `src.*` modules remain thin compatibility wrappers for existing local scripts. New projects should import from `adaptive_surrogate`.

Saved version-1 artifacts load with a warning and can be re-saved to the version-2 schema. Because artifacts use pickle, only load artifacts from trusted sources.
