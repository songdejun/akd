import akd.config as cfg_mod


def _write(path, text):
    path.write_text(text)


def test_defaults(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    cfg_mod.reset_config()
    cfg = cfg_mod.load()
    assert cfg.bench.rounds == 3
    assert cfg.dispatch.min_improvement == 0.03


def test_project_akd_toml(tmp_path, monkeypatch):
    _write(tmp_path / "akd.toml", "[bench]\nrounds = 7\n[store]\ndir = \"/tmp/x\"\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    cfg_mod.reset_config()
    cfg = cfg_mod.load()
    assert cfg.bench.rounds == 7
    assert cfg.store.dir == "/tmp/x"


def test_pyproject_tool_akd(tmp_path, monkeypatch):
    _write(
        tmp_path / "pyproject.toml",
        "[tool.akd.dispatch]\nmin_improvement = 0.10\n[tool.other]\nx = 1\n",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    cfg_mod.reset_config()
    assert cfg_mod.load().dispatch.min_improvement == 0.10


def test_project_beats_user(tmp_path, monkeypatch):
    home = tmp_path / "home" / ".config" / "akd"
    home.mkdir(parents=True)
    _write(home / "config.toml", "[bench]\nrounds = 5\nbudget_ms = 11.0\n")
    _write(tmp_path / "akd.toml", "[bench]\nrounds = 7\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    cfg_mod.reset_config()
    cfg = cfg_mod.load()
    assert cfg.bench.rounds == 7        # project wins
    assert cfg.bench.budget_ms == 11.0  # user keys not overridden survive


def test_env_overrides_cache_dir(tmp_path, monkeypatch):
    _write(tmp_path / "akd.toml", "[store]\ndir = \"/from/file\"\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("AKD_CACHE_DIR", "/from/env")
    cfg_mod.reset_config()
    assert cfg_mod.load().store.dir == "/from/env"


def test_env_debug_flag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for value, expected in [("0", False), ("true", True), ("yes", True), ("off", False)]:
        monkeypatch.setenv("AKD_DEBUG", value)
        cfg_mod.reset_config()
        assert cfg_mod.load().log.debug is expected, value


def test_unparseable_config_is_ignored(tmp_path, monkeypatch):
    _write(tmp_path / "akd.toml", "not [valid toml\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    cfg_mod.reset_config()
    assert cfg_mod.load().bench.rounds == 3
