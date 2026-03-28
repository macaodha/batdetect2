# Development and contribution

Thanks for your interest in improving batdetect2.

## Ways to contribute

- Report bugs and request features on
  [GitHub Issues](https://github.com/macaodha/batdetect2/issues)
- Improve docs by opening pull requests with clearer examples, fixes, or
  missing workflows
- Contribute code for models, data handling, evaluation, or CLI workflows

## Basic contribution workflow

1. Open an issue (or comment on an existing one) so work is visible.
2. Create a branch for your change.
3. Run checks locally before opening a PR:

```bash
just check
just docs
```

4. Open a pull request with a clear summary of what changed and why.

## Development environment

Use `uv` for dependency and environment management.

```bash
uv sync
```

For more setup details, see {doc}`../getting_started`.
