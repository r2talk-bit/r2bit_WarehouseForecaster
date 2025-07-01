# Previsor de Estoque para Armazém

Uma poderosa aplicação de previsão de séries temporais para gestão de estoque de armazéns, construída com Streamlit e o modelo de previsão Toto.

## Visão Geral

O Previsor de Estoque para Armazém é uma aplicação web que permite aos usuários carregar dados históricos de vendas/estoque e gerar previsões precisas para períodos futuros. A aplicação utiliza o modelo Toto, um modelo de previsão de séries temporais de última geração da Datadog, para fornecer previsões confiáveis.

## Funcionalidades

- **Interface fácil de usar**: Carregue seus dados e obtenha previsões com apenas alguns cliques
- **Visualização interativa**: Veja os resultados da previsão em tabelas e gráficos
- **Horizonte de previsão configurável**: Escolha quantos dias no futuro você deseja prever
- **Dados de exemplo incluídos**: Experimente a aplicação com dados de amostra incluídos
- **Processamento seguro**: Seus dados são processados com segurança e não são armazenados

## Requisitos

- Python 3.8+
- Dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
   ```
   git clone <repository-url>
   cd r2bit_WarehouseForecaster
   ```

2. Crie um ambiente virtual (recomendado):
   ```
   python -m venv venv
   ```

3. Ative o ambiente virtual:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```

## Executando a Aplicação

Inicie a aplicação Streamlit com:

```
streamlit run streamlit_app.py
```

A aplicação estará disponível em `http://localhost:8501` por padrão.

## Formato dos Dados

Seu arquivo CSV de entrada deve:
- Usar ponto e vírgula (`;`) como separador
- Usar vírgula (`,`) como separador decimal
- Conter duas colunas:
  - `DATE`: Formatada como DD/MM/AAAA
  - `VALUE`: Valores numéricos representando quantidades de vendas ou estoque

Exemplo:
```
DATE;VALUE
01/01/2023;120
02/01/2023;135
03/01/2023;142
...
```

## Configuração

A aplicação permite configurar:

- **Duração da Previsão**: O número de dias para prever (7-90 dias)

## Detalhes do Modelo

A previsão é alimentada pelo modelo Toto da Datadog, que é um modelo de previsão de séries temporais baseado em transformers. O pipeline do modelo:

1. Carrega o modelo Toto
2. Processa os dados de entrada
3. Executa a previsão usando TotoForecaster
4. Retorna a previsão como uma string CSV

## Segurança

- A aplicação executa validação de domínio para garantir que seja acessível apenas a partir de domínios autorizados
- Os dados são processados em memória e não são armazenados
- Nenhuma chamada de API externa é feita com seus dados

## Exemplo de Uso

1. Abra a aplicação em seu navegador web
2. Use os dados de exemplo fornecidos ou carregue seu próprio arquivo CSV
3. Defina a duração da previsão desejada usando o controle deslizante
4. Clique em "Gerar Previsão"
5. Veja os resultados e baixe a previsão como um arquivo CSV

## Solução de Problemas

Se você encontrar problemas:

1. Certifique-se de que seu arquivo CSV segue o formato necessário
2. Verifique se seus dados estão ordenados cronologicamente
3. Verifique se você tem todas as dependências necessárias instaladas

## Licença

Licença MIT

## Contato

Para suporte ou consultas, entre em contato com [R2Talk](https://waapp.r2talk.com.br).

---

# Warehouse Forecaster

A powerful time series forecasting application for warehouse inventory management, built with Streamlit and the Toto forecasting model.

## Overview

Warehouse Forecaster is a web application that allows users to upload historical sales/inventory data and generate accurate forecasts for future periods. The application uses the Toto model, a state-of-the-art time series forecasting model from Datadog, to provide reliable predictions.

## Features

- **Easy-to-use interface**: Upload your data and get forecasts with just a few clicks
- **Interactive visualization**: View your forecast results as both tables and charts
- **Configurable forecast horizon**: Choose how many days into the future you want to forecast
- **Example data included**: Try the application with included sample data
- **Secure processing**: Your data is processed securely and not stored

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd r2bit_WarehouseForecaster
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

Start the Streamlit app with:

```
streamlit run streamlit_app.py
```

The application will be available at `http://localhost:8501` by default.

## Data Format

Your input CSV file must:
- Use semicolon (`;`) as the separator
- Use comma (`,`) as the decimal separator
- Contain two columns:
  - `DATE`: Formatted as DD/MM/YYYY
  - `VALUE`: Numeric values representing sales or inventory quantities

Example:
```
DATE;VALUE
01/01/2023;120
02/01/2023;135
03/01/2023;142
...
```

## Configuration

The application allows you to configure:

- **Forecast Length**: The number of days to forecast (7-90 days)

## Model Details

The forecasting is powered by the Toto model from Datadog, which is a transformer-based time series forecasting model. The model pipeline:

1. Loads the Toto model
2. Processes the input data
3. Runs prediction using TotoForecaster
4. Returns the forecast as a CSV string

## Security

- The application runs domain validation to ensure it's only accessible from authorized domains
- Data is processed in-memory and not stored
- No external API calls are made with your data

## Example Usage

1. Open the application in your web browser
2. Use the provided example data or upload your own CSV file
3. Set the desired forecast length using the slider
4. Click "Generate Forecast"
5. View the results and download the forecast as a CSV file

## Troubleshooting

If you encounter issues:

1. Ensure your CSV file follows the required format
2. Check that your data is chronologically ordered
3. Verify that you have all required dependencies installed

## License

MIT License

## Contact

For support or inquiries, please contact [R2Talk](https://waapp.r2talk.com.br).





