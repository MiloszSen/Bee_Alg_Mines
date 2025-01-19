// script.js

// Przykładowa funkcja, która może zastąpić 
// kod z inline <script> w index.html
function runAlgorithm() {
    const formData = new FormData(document.getElementById("algoForm"));
    
    fetch("/run-algorithm", {
      method: "POST",
      body: formData
    })
    .then(response => response.json())
    .then(data => {
      if(data.error) {
        alert(data.error);
        return;
      }
      document.getElementById("profitLabel").innerText = 
        "Najlepszy osiągnięty zysk: " + data.bestProfit.toFixed(2);

      // Wyświetlenie całkowitego czasu wykonania algorytmu
      document.getElementById("timeLabel").innerText = 
        "Całkowity czas wykonania algorytmu: " + data.algTimeAll.toFixed(3) + " s";

      // Opcjonalnie: można tutaj wywołać funkcję do dalszej obróbki danych,
      // np. rysowanie grafu lub animacji
      // drawGraphAnimation(data.routeEdges);
    })
    .catch(err => console.error(err));
}

// Podłączenie obsługi submit formularza
document.getElementById("algoForm").addEventListener("submit", function(e) {
    e.preventDefault();
    runAlgorithm();
});
  