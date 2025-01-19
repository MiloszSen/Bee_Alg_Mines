
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

      
      document.getElementById("timeLabel").innerText = 
        "Całkowity czas wykonania algorytmu: " + data.algTimeAll.toFixed(3) + " s";

      
    })
    .catch(err => console.error(err));
}

document.getElementById("algoForm").addEventListener("submit", function(e) {
    e.preventDefault();
    runAlgorithm();
});
  
