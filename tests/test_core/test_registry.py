"""Tests for the Registry and SimpleRegistry classes.

Covers:
- SimpleRegistry: registration, retrieval, and membership checks.
- Registry: decorator-based registration, config type tracking,
  discriminator-based dispatch, and error handling.
"""

from typing import Literal

import pytest
from pydantic import BaseModel

from batdetect2.core.registries import Registry, SimpleRegistry


class TestSimpleRegistry:
    def test_register_and_get(self):
        """Registered objects can be retrieved by name."""
        registry = SimpleRegistry("test")

        @registry.register("my_item")
        def item():
            return 42

        assert registry.get("my_item")() == 42

    def test_register_returns_original_object(self):
        """The register decorator returns the decorated object unchanged."""
        registry = SimpleRegistry[int]("test")

        @registry.register("x")
        def fn() -> int:
            return 7

        assert fn() == 7

    def test_has_returns_true_for_registered_name(self):
        """has() returns True for a name that was registered."""
        registry = SimpleRegistry("test")
        registry.register("present")(lambda: None)
        assert registry.has("present") is True

    def test_has_returns_false_for_unknown_name(self):
        """has() returns False for a name that was never registered."""
        registry = SimpleRegistry("test")
        assert registry.has("absent") is False

    def test_get_raises_for_unknown_name(self):
        """get() raises KeyError for an unregistered name."""
        registry = SimpleRegistry("test")
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_register_overwrites_existing_entry(self):
        """Re-registering the same name replaces the previous entry."""
        registry = SimpleRegistry("test")
        registry.register("key")(lambda: 1)
        registry.register("key")(lambda: 2)
        assert registry.get("key")() == 2

    def test_multiple_items_registered_independently(self):
        """Multiple items can be registered without interfering."""
        registry = SimpleRegistry("test")
        registry.register("a")(lambda: "a")
        registry.register("b")(lambda: "b")
        assert registry.get("a")() == "a"
        assert registry.get("b")() == "b"


class TestRegistryRegister:
    def test_register_makes_factory_callable_via_build(self):
        """A registered factory is reachable through build()."""

        class DummyConfig(BaseModel):
            name: Literal["dummy"] = "dummy"
            value: int = 0

        class DummyOutput:
            def __init__(self, config: DummyConfig):
                self.config = config

        registry: Registry[DummyOutput, []] = Registry("test")
        registry.register(DummyConfig)(lambda c: DummyOutput(c))
        result = registry.build(DummyConfig(value=3))
        assert isinstance(result, DummyOutput)
        assert result.config.value == 3

    def test_register_makes_config_type_retrievable(self):
        """A registered config type is reachable through get_config_type()."""

        class DummyConfig(BaseModel):
            name: Literal["dummy"] = "dummy"
            value: int = 0

        registry: Registry[object, []] = Registry("test")
        registry.register(DummyConfig)(lambda c: c)
        assert registry.get_config_type("dummy") is DummyConfig

    def test_register_raises_when_discriminator_field_missing(self):
        """ValueError is raised if config has no discriminator field."""

        class ConfigWithoutDiscriminator(BaseModel):
            unrelated_field: str = "hello"

        registry: Registry[object, []] = Registry("test")
        with pytest.raises(ValueError, match="'name' field"):
            registry.register(ConfigWithoutDiscriminator)(lambda c: c)

    def test_register_raises_when_discriminator_is_not_string(self):
        """ValueError is raised if the discriminator default is not a str."""

        class ConfigWithNonStringDiscriminator(BaseModel):
            name: int = 42

        registry: Registry[object, []] = Registry("test")
        with pytest.raises(ValueError, match="'name' field must be a string"):
            registry.register(ConfigWithNonStringDiscriminator)(lambda c: c)

    def test_register_uses_custom_discriminator_field(self):
        """Registry respects a non-default discriminator field name."""

        class FormatConfig(BaseModel):
            format: Literal["fmt"] = "fmt"

        registry: Registry[object, []] = Registry(
            "test", discriminator="format"
        )
        registry.register(FormatConfig)(lambda c: c)
        assert registry.get_config_type("fmt") is FormatConfig

    def test_register_decorator_returns_original_function(self):
        """The register decorator returns the wrapped function unchanged."""

        class DummyConfig(BaseModel):
            name: Literal["dummy"] = "dummy"
            value: int = 0

        class DummyOutput:
            def __init__(self, config: DummyConfig):
                self.config = config

        registry: Registry[DummyOutput, []] = Registry("test")

        def factory(config: DummyConfig) -> DummyOutput:
            return DummyOutput(config)

        returned = registry.register(DummyConfig)(factory)
        assert returned is factory


class TestRegistryConfigTypes:
    def test_get_config_types_returns_all_registered_types(self):
        """get_config_types() returns every registered config class."""

        class DummyConfig(BaseModel):
            name: Literal["dummy"] = "dummy"

        class AnotherConfig(BaseModel):
            name: Literal["another"] = "another"

        registry: Registry[object, []] = Registry("test")
        registry.register(DummyConfig)(lambda c: c)
        registry.register(AnotherConfig)(lambda c: c)
        config_types = registry.get_config_types()
        assert DummyConfig in config_types
        assert AnotherConfig in config_types

    def test_get_config_types_empty_when_nothing_registered(self):
        """get_config_types() returns empty tuple for a fresh registry."""
        registry: Registry[object, []] = Registry("test")
        assert registry.get_config_types() == ()

    def test_get_config_type_returns_correct_class(self):
        """get_config_type() returns the class registered under a key."""

        class DummyConfig(BaseModel):
            name: Literal["dummy"] = "dummy"

        registry: Registry[object, []] = Registry("test")
        registry.register(DummyConfig)(lambda c: c)
        assert registry.get_config_type("dummy") is DummyConfig

    def test_get_config_type_raises_for_unknown_key(self):
        """get_config_type() raises ValueError for an unregistered key."""
        registry: Registry[object, []] = Registry("test")
        with pytest.raises(
            ValueError, match="No config type with name 'unknown'"
        ):
            registry.get_config_type("unknown")

    def test_get_config_type_error_message_lists_existing_keys(self):
        """ValueError message includes the names of registered keys."""

        class DummyConfig(BaseModel):
            name: Literal["dummy"] = "dummy"

        registry: Registry[object, []] = Registry("test")
        registry.register(DummyConfig)(lambda c: c)
        with pytest.raises(ValueError, match="dummy"):
            registry.get_config_type("missing")


class TestRegistryBuild:
    def test_build_dispatches_to_correct_factory(self):
        """build() calls the factory registered for the config's discriminator."""

        class DummyConfig(BaseModel):
            name: Literal["dummy"] = "dummy"
            value: int = 0

        class DummyOutput:
            def __init__(self, config: DummyConfig):
                self.config = config

        registry: Registry[DummyOutput, []] = Registry("test")
        registry.register(DummyConfig)(lambda c: DummyOutput(c))

        config = DummyConfig(value=99)
        result = registry.build(config)

        assert isinstance(result, DummyOutput)
        assert result.config.value == 99

    def test_build_dispatches_to_correct_factory_among_multiple(self):
        """build() picks the right factory when several are registered."""

        class DummyConfig(BaseModel):
            name: Literal["dummy"] = "dummy"

        class AnotherConfig(BaseModel):
            name: Literal["another"] = "another"

        class DummyOutput:
            def __init__(self, config: DummyConfig):
                self.config = config

        class AnotherOutput:
            def __init__(self, config: AnotherConfig):
                self.config = config

        registry: Registry[object, []] = Registry("test")
        registry.register(DummyConfig)(lambda c: DummyOutput(c))
        registry.register(AnotherConfig)(lambda c: AnotherOutput(c))

        assert isinstance(registry.build(DummyConfig()), DummyOutput)
        assert isinstance(registry.build(AnotherConfig()), AnotherOutput)

    def test_build_raises_not_implemented_for_unregistered_format(self):
        """build() raises NotImplementedError for an unregistered discriminator."""
        registry: Registry[object, []] = Registry("test")

        class UnknownConfig(BaseModel):
            name: Literal["unknown"] = "unknown"

        with pytest.raises(NotImplementedError, match="'unknown'"):
            registry.build(UnknownConfig())

    def test_build_passes_config_to_factory(self):
        """build() passes the exact config object through to the factory."""

        class DummyConfig(BaseModel):
            name: Literal["dummy"] = "dummy"
            value: int = 0

        registry: Registry[DummyConfig, []] = Registry("test")
        received: list[DummyConfig] = []
        registry.register(DummyConfig)(lambda c: received.append(c) or c)

        config = DummyConfig(value=7)
        registry.build(config)

        assert received == [config]

    def test_build_uses_custom_discriminator_field(self):
        """build() resolves the factory using the configured discriminator."""

        class FormatConfig(BaseModel):
            format: Literal["fmt"] = "fmt"

        registry: Registry[str, []] = Registry("test", discriminator="format")
        registry.register(FormatConfig)(lambda c: "fmt_result")

        assert registry.build(FormatConfig()) == "fmt_result"

    def test_build_error_message_includes_registry_name(self):
        """NotImplementedError message names the registry for easier debugging."""
        registry: Registry[object, []] = Registry("my_registry")

        class UnknownConfig(BaseModel):
            name: Literal["ghost"] = "ghost"

        with pytest.raises(NotImplementedError, match="my_registry"):
            registry.build(UnknownConfig())
