def _patch_dify_plugin():
    """Patch older dify_plugin versions missing LLMResultChunkWithStructuredOutput."""
    try:
        import dify_plugin.entities.model.llm as _llm
        if hasattr(_llm, "LLMResultChunkWithStructuredOutput"):
            return
        from pydantic import BaseModel
        _LLMResultChunk = _llm.LLMResultChunk

        class LLMStructuredOutput(BaseModel):
            structured_output: dict = {}

        class LLMResultChunkWithStructuredOutput(_LLMResultChunk):
            structured_output: dict = {}

        class LLMResultWithStructuredOutput(_llm.LLMResult):
            structured_output: dict = {}

        _llm.LLMStructuredOutput = LLMStructuredOutput
        _llm.LLMResultChunkWithStructuredOutput = LLMResultChunkWithStructuredOutput
        _llm.LLMResultWithStructuredOutput = LLMResultWithStructuredOutput
    except Exception:
        pass


_patch_dify_plugin()

from dify_plugin import Plugin, DifyPluginEnv
import logging

logging.basicConfig(level=logging.INFO)

plugin = Plugin(DifyPluginEnv())

if __name__ == "__main__":
    plugin.run()
