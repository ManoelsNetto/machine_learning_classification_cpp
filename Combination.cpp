#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <numeric>

using namespace std;

// Função recursiva otimizada para gerar combinações de valores
bool find_combinations(vector<double>& valores, vector<double>& current_combination, int start, double soma_desejada, double& closest_sum, vector<double>& best_combination) {
    double current_sum = accumulate(current_combination.begin(), current_combination.end(), 0.0);

    // Se a soma atual exceder a soma desejada, poda a busca
    if (current_sum > soma_desejada + 1e-9) {
        return false; 
    }

    // Se a soma atual for exata, atualiza e retorna
    if (fabs(current_sum - soma_desejada) < 1e-9) {
        closest_sum = current_sum;
        best_combination = current_combination;
        return true; // Encontrou a combinação exata
    }

    // Se a soma atual for mais próxima da soma desejada, atualiza os melhores valores
    if (closest_sum == -1 || abs(current_sum - soma_desejada) < abs(closest_sum - soma_desejada)) {
        closest_sum = current_sum;
        best_combination = current_combination;
    }

    // Itera para gerar combinações de valores
    for (int i = start; i < valores.size(); i++) {
        current_combination.push_back(valores[i]);
        if (find_combinations(valores, current_combination, i + 1, soma_desejada, closest_sum, best_combination)) {
            return true; // Parar a busca se encontrar uma combinação exata
        }
        current_combination.pop_back();
    }

    return false;
}

int main() {
    vector<double> values;
    double wanted_sum;
    double closest_sum = -1;
    vector<double> best_combination;
    vector<double> current_combination;

    string line;
    
    cout << "Quais os dados:" << endl;
    
    while (getline(cin, line) && !line.empty()) {
        double value = stod(line);  // Conversão direta para double
        values.push_back(value);
    }

    cout << "Qual a soma desejada: ";
    cin >> wanted_sum;

    // Ordena os valores para facilitar a poda
    sort(values.begin(), values.end());

    find_combinations(values, current_combination, 0, wanted_sum, closest_sum, best_combination);

    cout << "Melhor combinacao: ";
    for (double val : best_combination) {
        cout << val << " ";
    }
    cout << "\nSoma mais proxima: " << fixed << closest_sum << endl;
    cout << "Diferenca com a soma desejada: " << fixed << closest_sum - wanted_sum << endl;

    return 0;
}