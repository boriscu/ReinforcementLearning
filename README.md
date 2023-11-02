# Pravila za Repozitorijum - Samoobučavajući i Adaptivni Algoritmi

## Opšta Pravila

1. **Struktura Repozitorijuma**: Svaka skripta predstavlja poseban domaći zadatak. 
2. **Rad na Zadatku**: Svako radi svoj deo zadatka posebno i odvaja ga sa komentarima u formatu:
    ```python
    # Zadatak _brojZadatka_ : _tekstZadatka_
    ```

## Git Workflow

1. **Kloniranje Repozitorijuma**:
    ```bash
    git clone <url-repozitorijuma>
    cd <ime-repozitorijuma>
    ```

2. **Kreiranje i Prebacivanje na Novu Granu za Zadatak**:
    ```bash
    git checkout -b zadatak/imeZadatka
    ```

3. **Rad na Zadatku**: 
   - Pišite i menjajte kod kako je potrebno za vaš zadatak.
   - Redovno commit-ujte vaše promene sa smislenim porukama.

4. **Dodavanje i Commit-ovanje Promena**:
    ```bash
    git add .
    git commit -m "Opis šta je urađeno u zadatku"
    ```

5. **Push-ovanje Grane na Repozitorijum**:
    ```bash
    git push origin zadatak/imeZadatka
    ```

6. **Kreiranje Pull Request-a (PR)**:
   - Idite na GitHub stranicu repozitorijuma.
   - Trebalo bi da vidite opciju za kreiranje pull request-a za vašu granu.
   - Postavite naslov i opis koji jasno objašnjavaju šta ste uradili.
   - Zatražite review od dva člana tima.

7. **Pregled i Odobravanje PR-a**:
   - Članovi tima pregledaju PR, ostavljaju komentare i sugestije ako je potrebno.
   - Ako su potrebne izmene, vršite ih na istoj grani i ponovite korake 4 i 5.
   - Nakon što dva člana tima odobre PR, može se spojiti (merge) sa main granom.

8. **Spajanje Grane sa Main Granom**:
   - Ovo možete uraditi direktno na GitHub stranici nakon odobravanja PR-a.
   - Kliknite na "Merge pull request" i zatim "Confirm merge".

9. **Ažuriranje Lokalne Main Grane**:
    ```bash
    git checkout main
    git pull origin main
    ```

10. **Brisanje Grane Nakon Spajanja**:
    - Lokalno:
      ```bash
      git branch -d zadatak/imeZadatka
      ```
    - Na GitHub-u:
      - Grana bi trebalo automatski da se obriše nakon spajanja, ukoliko se to ne desi molim vas da je vi obrišete
   
Pratite ova uputstva kako biste osigurali uredan i efikasan rad na projektu.
