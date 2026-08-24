def test_version_exposed():
    import vector_service
    assert isinstance(vector_service.__version__, str)
    assert len(vector_service.__version__.split(".")) == 3
