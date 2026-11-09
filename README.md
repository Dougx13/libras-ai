# Libras AI — ASP.NET Core MVC

Aplicação web de soletração em Libras. A câmera rastreia até duas mãos ao mesmo tempo. O modo estático reconhece 21 poses; o modo separado de movimento compara exemplos gravados para Ç, H, J, K, X, Y e Z. K também é reconhecida pela pose estática.

## Abrir no Visual Studio 2022

1. Abra `LibrasAI.sln` no Visual Studio 2022.
2. Selecione o perfil `https` e pressione **F5**.
3. Permita o uso da câmera no navegador e selecione **Ativar câmera**. O reconhecimento das letras começa automaticamente.
4. Mantenha cada pose parada por um instante. A letra estática só é inserida depois de uma sequência consistente de quadros com confiança e diferença claras para a segunda opção. Para movimentos de Ç, H, J, K, X, Y e Z, ative o botão **Letras com movimento**. Nesse modo, a classificação estática pausa. Escolha letra e mão, clique **Gravar exemplo** e faça o sinal completo quando o painel avisar que está pronto. A letra fica habilitada após 5 exemplos para aquela mão; grave 20 ou mais exemplos variados para melhorar a comparação. Desative o modo para voltar às letras paradas.
5. Os exemplos ficam guardados localmente no navegador (IndexedDB). Use **Baixar exemplos JSON** para guardar uma cópia. Para incluir os exemplos no Git, substitua `wwwroot/models/libras-dynamic-samples.json` pelo JSON exportado. Para carregar exemplos de outro navegador, use **Importar exemplos JSON**.

O projeto usa `net9.0`. O Visual Studio precisa ter a carga de trabalho **ASP.NET e desenvolvimento Web** e suporte ao SDK .NET 9 (Visual Studio 2022 17.12 ou posterior).

## Executar pelo terminal

```bash
dotnet restore
dotnet run
```

Abra a URL HTTPS exibida no terminal. O navegador só libera a câmera em HTTPS ou em `localhost`.

## Como funciona

- `Controllers/HomeController.cs` e `Views/Home/Index.cshtml` implementam a rota MVC e a interface.
- `wwwroot/js/app.js` usa MediaPipe Hand Landmarker para localizar a mão, normaliza os 21 pontos e executa o Random Forest localmente no navegador.
- `wwwroot/models/libras-static-v1.json` contém o classificador já treinado. A inferência não depende de Python, de uma API externa ou de exemplos cadastrados pelo usuário.
- O mesmo `app.js` grava sequências de pontos e compara o gesto completo com exemplos por DTW (Dynamic Time Warping). Isso é um reconhecedor temporal por exemplos, não um modelo neural treinado; os dados podem ser exportados/importados em JSON.
- `wwwroot/models/libras-dynamic-samples.json` é o arquivo inicial de exemplos temporais. Ele começa vazio; é necessário gravar ou importar exemplos para reconhecer dinamicamente Ç, H, J, K, X, Y e Z.
- `wwwroot/img/alfabeto/` contém recortes ilustrativos para A–Z e Ç, com indicação visual das letras que o classificador reconhece e das que são apenas referência.
- `wwwroot/css/styles.css` contém os estilos.
- O vídeo da câmera permanece no navegador. A biblioteca de visão e o modelo do MediaPipe são carregados de CDNs; a primeira execução requer internet.

## Escopo e limitações

O classificador estático cobre **A, B, C, D, E, F, G, I, K, L, M, N, O, P, Q, R, S, T, U, V e W**. A câmera suaviza três quadros antes de classificar e usa extensão, curvatura e distância entre pontas dos dedos para desempatar formatos de mão mais distintos. Para S, também compara a posição do polegar sobre a palma com as articulações dos dedos fechados. O comparador temporal aceita Ç, H, J, K, X, Y e Z depois de reunir pelo menos 5 exemplos por letra e mão. A comparação temporal suaviza tremores, dá mais peso à trajetória que à forma da mão e tolera variações moderadas na velocidade. O reconhecimento rejeita previsões incertas e exige consistência ao longo do tempo; isso pode levar alguns instantes a confirmar uma pose. O modo estático e o modo de movimento não atuam ao mesmo tempo. O resultado depende dos exemplos gravados e precisa ser avaliado com sinais reais para medir a precisão. O texto resultante é soletração; o sistema não reconhece sinais de vocabulário nem converte Libras em português gramatical.

O desempenho depende de iluminação, posição, enquadramento e semelhança com os exemplos do conjunto de treino. A interface orienta quando a mão está cortada, distante ou ainda instável. O modelo obteve F1 macro 0,939 em avaliação holdout relatada pelo projeto de origem; isso não é garantia de precisão para todos os usuários e câmeras. O modelo agrega dados de imagens de alfabeto e de outro conjunto de letras em Libras. Os recortes do guia são referências para estudo, não imagens de treino. A seção de vocabulário encaminha para exemplos ilustrados externos de lei, e-mail, chuva, calor, frio, Libras, triste e amigo; esses sinais ainda não são classificados pela câmera. Consulte as atribuições e licenças em `THIRD-PARTY-NOTICES.txt`.

## Tecnologias

- ASP.NET Core MVC / C#
- HTML, CSS e JavaScript
- MediaPipe Tasks Vision no navegador
- Random Forest exportado para JSON para inferência local
