const $ = (s) => document.querySelector(s);

async function search() {
  const q = $("#q").value.trim();
  const k = parseInt($("#k").value, 10) || 5;
  const min = parseFloat($("#min").value) || 0.0;
  if (!q) { alert("Empty query"); return; }

  const url = `http://127.0.0.1:8000/search?q=${encodeURIComponent(q)}&top_k=${k}&min_score=${min}`;
  const res = await fetch(url);
  const data = await res.json();

  const ul = $("#results");
  ul.innerHTML = "";
  if (!Array.isArray(data) || data.length === 0) {
    ul.innerHTML = "<li>No matches.</li>";
    return;
  }

  data.forEach((it) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="meta"><b>ID:</b> ${it.id} &nbsp; | &nbsp; <b>score:</b> ${it.score.toFixed(3)}</div>
      <div>${it.text}</div>
    `;
    ul.appendChild(li);
  });
}

$("#go").addEventListener("click", search);
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter") search(); });
