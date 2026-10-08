"""mdBook bridge to Griffe's Markdown renderer for curated API comments."""

import json
import os
import re
import sys
from pathlib import Path

import yaml
from griffe import GriffeLoader, Parser
from griffe2md import render_object_docs


def chapters(items):
    for item in items:
        if "Chapter" in item:
            chapter = item["Chapter"]
            yield chapter
            yield from chapters(chapter["sub_items"])


def preprocess(context, book):
    root = Path(context["root"])
    loader = GriffeLoader(
        search_paths=[root], docstring_parser=Parser.google, allow_inspection=False
    )
    loader.load("rsikit")
    loader.load("research")
    loader.resolve_aliases()
    pages = list(chapters(book["items"]))
    anchors = {}
    revision = os.environ.get("GITHUB_SHA", "main")

    for page in pages:

        def render(match):
            obj = loader.modules_collection[match[1]]
            options = {
                "heading_level": 2,
                "show_signature_annotations": True,
                "members_order": "source",
                "inherited_members": False,
                "summary": False,
                **(yaml.safe_load(match[2]) or {}),
            }
            if isinstance(options.get("members"), list):
                for member in options["members"]:
                    if member not in obj.all_members:
                        raise ValueError(f"Unknown API member: {obj.path}.{member}")
            markdown = render_object_docs(obj, options)

            def heading(match):
                name = match[2]
                anchors[name] = page["path"]
                return f'<a id="{name}"></a>\n\n{match[0]}'

            markdown = re.sub(r"^(#{2,6}) `([\w.]+)`$", heading, markdown, flags=re.M)
            source = obj.filepath.relative_to(root)
            url = f"https://github.com/StuartFarmer/rsikit/blob/{revision}/{source}#L{obj.lineno}"
            return f"{markdown}\n[Source]({url})\n"

        page["content"] = re.sub(
            r"<!-- api: ([\w.]+)\n(.*?)\n-->", render, page["content"], flags=re.S
        )

    # Griffe emits symbolic type links; link documented objects and leave other
    # types as text instead of publishing dead links to builtins or private APIs.
    for page in pages:

        def crossref(match):
            label, name = match.groups()
            if name not in anchors:
                return label
            target = os.path.relpath(anchors[name], Path(page["path"]).parent)
            return f"[{label}]({target}#{name})"

        page["content"] = re.sub(r"\[([^\[\]\n]*)\]\(#([\w.]+)\)", crossref, page["content"])
    return book


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "supports":
        sys.exit(0 if sys.argv[2] == "html" else 1)
    context, book = json.load(sys.stdin)
    json.dump(preprocess(context, book), sys.stdout)
