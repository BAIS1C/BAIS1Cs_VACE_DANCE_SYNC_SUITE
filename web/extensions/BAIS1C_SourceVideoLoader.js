/* BAIS1C VACE Suite – metadata display widget */
import { app } from "/scripts/app.js";

app.registerExtension({
  name: "BAIS1C.SourceVideoLoader",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "BAIS1C_SourceVideoLoader") return;

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (msg) {
      onExecuted?.apply(this, arguments);

      // Remove previous panel if it exists
      this.metaPanel?.remove();
      this.metaPanel = null;

      const text = msg?.ui_info?.[0];
      if (!text) return;

      const div = document.createElement("div");
      div.style.cssText = `
        border:1px solid #444;
        background:#222;
        color:#eee;
        padding:6px 8px;
        margin-top:4px;
        font:11px/1.2 monospace;
        white-space:pre-wrap;
        max-height:120px;
        overflow-y:auto;
      `;

      // Colour the Aspect line
      div.innerHTML = text
        .split("\n")
        .map(line =>
          line.startsWith("Aspect:")
            ? `<span style="color:${line.includes("✅") ? "#0f0" : "#ff0"}">${line}</span>`
            : line
        )
        .join("\n");

      this.addDOMWidget("meta_panel", "meta_panel", div);
      this.metaPanel = div;
    };
  },
});