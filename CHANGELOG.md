# nvforest 26.04.00 (8 Apr 2026)

### 🚨 Breaking Changes
* Make `optimize` return a new instance instead of mutating in-place by @dantegd in https://github.com/rapidsai/nvforest/pull/31
### 📖 Documentation
* Set up docs build by @hcho3 in https://github.com/rapidsai/nvforest/pull/35
* [Doc] Add BUILD.md by @hcho3 in https://github.com/rapidsai/nvforest/pull/41
* Remove remaining references to cuML by @hcho3 in https://github.com/rapidsai/nvforest/pull/47
* [Doc] Add tutorial by @hcho3 in https://github.com/rapidsai/nvforest/pull/48
* Address comments in rapidsai/nvforest#35 by @hcho3 in https://github.com/rapidsai/nvforest/pull/46
* Fix typos and syntax error in tutorial by @hcho3 in https://github.com/rapidsai/nvforest/pull/60
* add CHANGELOG.md by @jameslamb in https://github.com/rapidsai/nvforest/pull/79
### 🚀 New Features
* Add `optimize()` method for automatic layout and chunk_size tuning by @dantegd in https://github.com/rapidsai/nvforest/pull/30
### 🛠️ Improvements
* [CI] Enable more CI jobs by @csadorf in https://github.com/rapidsai/nvforest/pull/20
* [CI] Enable conda-python-tests and bump scikit-learn to 1.5 by @csadorf in https://github.com/rapidsai/nvforest/pull/21
* [CI ] Enable wheel-tests-cuforest and fix treelite linking by @csadorf in https://github.com/rapidsai/nvforest/pull/23
* Empty commit to trigger a build by @jameslamb in https://github.com/rapidsai/nvforest/pull/27
* Use main shared-workflows branch by @jameslamb in https://github.com/rapidsai/nvforest/pull/29
* remove unused dependencies by @jameslamb in https://github.com/rapidsai/nvforest/pull/25
* enforce tighter wheel-size limits in CI by @jameslamb in https://github.com/rapidsai/nvforest/pull/26
* wheel builds: react to changes in pip's handling of build constraints by @mmccarty in https://github.com/rapidsai/nvforest/pull/33
* Add instructions for CodeRabbit by @csadorf in https://github.com/rapidsai/nvforest/pull/36
* CI: Add agents instructions to change-files exclusion list. by @csadorf in https://github.com/rapidsai/nvforest/pull/37
* Update to 26.04 by @hcho3 in https://github.com/rapidsai/nvforest/pull/34
* Drop Python 3.10 support by @gforsyth in https://github.com/rapidsai/nvforest/pull/38
* expand CI-skipping logic, other small build changes by @jameslamb in https://github.com/rapidsai/nvforest/pull/39
* remove pip.conf migration code in CI scripts by @jameslamb in https://github.com/rapidsai/nvforest/pull/40
* Use GHA id-token for `sccache-dist` auth token by @trxcllnt in https://github.com/rapidsai/nvforest/pull/44
* refactor update-version.sh to handle new branching strategy by @jameslamb in https://github.com/rapidsai/nvforest/pull/42
* refactor: build wheels and conda packages using Python limited API by @gforsyth in https://github.com/rapidsai/nvforest/pull/43
* refactor(limited api): add explicit `wheel.py-api` to `pyproject.toml`
 by @gforsyth in https://github.com/rapidsai/nvforest/pull/50
* Update codebase to rename cuforest -> nvforest by @hcho3 in https://github.com/rapidsai/nvforest/pull/51
* Rename cuforest to nvforest in coderabbit config. by @csadorf in https://github.com/rapidsai/nvforest/pull/53
* Add README by @csadorf in https://github.com/rapidsai/nvforest/pull/55
* enable branch builds on merge by @jameslamb in https://github.com/rapidsai/nvforest/pull/56
* enable nightly tests by @jameslamb in https://github.com/rapidsai/nvforest/pull/58
* Add support for Python 3.14 by @gforsyth in https://github.com/rapidsai/nvforest/pull/49
* Prevent integer overflow by @hcho3 in https://github.com/rapidsai/nvforest/pull/63
* avoid packaging lib_treelite.{a,so} by @jameslamb in https://github.com/rapidsai/nvforest/pull/77
* update-version.sh: update usage docs, other small changes by @jameslamb in https://github.com/rapidsai/nvforest/pull/81
* trim docs environment by @jameslamb in https://github.com/rapidsai/nvforest/pull/80
* [CI] Enable devcontainer jobs by @hcho3 in https://github.com/rapidsai/nvforest/pull/82
* build wheels with CUDA 13.0.x, test wheels against mix of CTK versions, drop CUDA math libraries dependencies by @jameslamb in https://github.com/rapidsai/nvforest/pull/87
* mark 'packaging' as test-only dependency by @jameslamb in https://github.com/rapidsai/nvforest/pull/88
* Check input dimensions by @hcho3 in https://github.com/rapidsai/nvforest/pull/76
* Add release.yml for PR categorization by @AyodeAwe in https://github.com/rapidsai/nvforest/pull/97

## New Contributors
* @mmccarty made their first contribution in https://github.com/rapidsai/nvforest/pull/33

**Full Changelog**: https://github.com/rapidsai/nvforest/compare/v26.04.00a...release/26.04

# nvForest initial public development (March 2026)

Public development on nvForest began in March 2026.
