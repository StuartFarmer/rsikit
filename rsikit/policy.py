"""A generated policy value. Its executable implementation is private to RSIKit."""

import ast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class Policy(BaseModel):
    """Returned by generation; pass it directly to Run.evaluate()."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str = Field(min_length=1)
    summary: str = ""
    _implementation: str = PrivateAttr(default="")


class _PolicyResponse(BaseModel, extra="forbid"):
    """Wire format between the coding model and the generation boundary."""

    name: str = Field(min_length=1)
    summary: str = ""
    implementation: str = Field(min_length=1)

    def policy(self) -> Policy:
        tree = ast.parse(self.implementation)
        if not self.name.strip():
            raise ValueError("The generated policy needs a name")
        if not any(
            isinstance(node, ast.ClassDef) and node.name == "Solution" for node in tree.body
        ):
            raise ValueError("The generated policy must define a Solution class")
        policy = Policy(name=self.name, summary=self.summary)
        policy._implementation = self.implementation
        return policy
