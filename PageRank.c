
#include <stdio.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>

//# Directed graph (each unordered pair of nodes is saved once): Wiki-Vote.txt 
//# Wikipedia voting on promotion to administratorship (till January 2008).
//# Directed edge A->B means user A voted on B becoming Wikipedia administrator.
static int k=0;

int max=255; // variable global pour les prochaines utilisations
FILE* repertoire = NULL; // Déclaration d'un fichier

void AffcherLaListeDesPays(){
	
	repertoire = fopen("Wiki-Vote.txt","r");
	char line[255];
	printf("\t\t\t Affichage de donnees existent dans notre DATASET \n");
	while (fgets(line,max,repertoire)!= NULL) {
		   printf("	%s ",line);
    }	
}
int main(){
  /*************************** Affichage de données existent dans la DATASET  ***************************/
   AffcherLaListeDesPays();
  //FIN de l'affichage
  
  /*************************** TEMPS,VARIABLES ***************************/

  // Gardez une trace du temps d'exécution
  clock_t begin, end;
  double time_spent;
  begin = clock();
  int numthreads = 16;
  int granularity = 8;

  /******************* FICHIER OUVERT + NOMBRE DE NŒUDS/ARÊTS ********************/

  // Ouvrir la DATASET 
  char filename[]="Wiki-Vote.txt";
  FILE *fp;
  if((fp=fopen(filename,"r"))==NULL) {
    fprintf(stderr,"[Error] ne peut pas ouvrir le fichier");
    exit(1);
  }
  
  // Lire l'ensemble de données et obtenir le nombre de nœuds (n) et d'arêtes (e)
  int n, e;
  char ch;
  char str[100];
  ch = getc(fp);
  while((ch = fgetc(fp)) != EOF) {
    fgets(str, 100-1, fp);
    sscanf (str,"%*s %d %*s %d", &n, &e); //nombre de nœuds
    ch = getc(fp);
  }
  ungetc(ch,fp);
  
  // DEBUG : imprimez le nombre de nœuds et d'arêtes, ignorez tout le reste
  printf("\nDonnee du graphe :\n\n Sommet : %d, Arret: %d \n\n", n, e);  
  
  /************************* STRUCTURES RSE *****************************/
    
  /* Format de ligne clairsemé compressé :
      - Val vector : contient 1,0 si une arête existe dans une certaine ligne
      - Vecteur Col_ind : contient l'index de colonne de la valeur correspondante dans 'val'
      - Vecteur Row_ptr : pointe vers le début de chaque ligne dans 'col_ind'
  */
  float *val = calloc(e, sizeof(float));
  int *col_ind = calloc(e, sizeof(int));
  int *row_ptr = calloc(n+1, sizeof(int));
 
  // La première ligne commence toujours à la position 0
  row_ptr[0] = 0;

  int fromnode, tonode;
  int cur_row = 0;
  int i = 0;
  int j = 0;
  // Éléments pour la ligne
  int elrow = 0;
  // Nombre cumulé d'éléments
  int curel = 0;
  
  while(!feof(fp)){
    
    fscanf(fp,"%d%d",&fromnode,&tonode);
       
    if (fromnode > cur_row) { // change the row
      curel = curel + elrow;
      for (k = cur_row + 1; k <= fromnode; k++) {
        row_ptr[k] = curel;
      }
      elrow = 0;
      cur_row = fromnode;
    }
    val[i] = 1.0;
    col_ind[i] = tonode;
    elrow++;
    i++;
  }
  row_ptr[cur_row+1] = curel + elrow - 1;

  // Corriger la stochastisation
  int out_link[n];
  for(i=0; i<n; i++) {
    out_link[i] = 0;
  }


  int rowel = 0;
  for(i=0; i<n; i++){
        if (row_ptr[i+1] != 0) {
          rowel = row_ptr[i+1] - row_ptr[i];
          out_link[i] = rowel;
        }
   }

    
  int curcol = 0;
  for(i=0; i<n; i++) {
    rowel = row_ptr[i+1] - row_ptr[i];
    for (j=0; j<rowel; j++) {
      val[curcol] = val[curcol] / out_link[i];
      curcol++;
    }
  }

 
  /******************* INITIALISATION DE P, FACTEUR D'AMORTISSEMENT ************************/

  // Définit le facteur d'amortissement 'd'
  float d = 0.85;
  
  // Initialise le vecteur p[]
  float p[n];
  for(i=0; i<n; i++){
    p[i] = 1.0/n;
  }
  
  // Définit la condition de bouclage et le nombre d'itérations 'k'
  int looping = 1;
  int k = 0;

  // Définit 'parallel' en fonction du nombre de threads
  int parallel = 0;
  if (numthreads >= 2) {
          parallel = 1;
  }

  // Initialise p_new comme un vecteur de n 0.0 cellules    for(i=0; i<n; i++){
  float p_new[n];
  
  /*************************** BOUCLE DE PageRank  **************************/

  while (looping){

    // Initialize p_new as a vector of n 0.0 cells
    for(i=0; i<n; i++){
      p_new[i] = 0.0;
    }
    
    int rowel = 0;
    int curcol = 0;
    
	// Algorithme de page rank modifié + parallélisation
    #pragma omp parallel for schedule(static) if(parallel) num_threads(numthreads)
    for(i=0; i<n; i = i + granularity){
      rowel = row_ptr[i+1] - row_ptr[i];
      for (j=0; j<rowel; j++) {
        p_new[col_ind[curcol]] = p_new[col_ind[curcol]] + val[curcol] * p[i];
        curcol++;
      }
    }

	// Ajustement pour gérer les éléments pendants  
    for(i=0; i<n; i++){
      p_new[i] = d * p_new[i] + ((1.0 - d) / n);
    }
       
	// TERMINATION : vérifie si nous devons nous arrêter
    float error = 0.0;
    for(i=0; i<n; i++){
      error =  error + fabs(p_new[i] - p[i]);
    }
	//si deux instances consécutives du vecteur pagerank sont presque identiques, stop   
    if (error < 0.000001){
      looping = 0;
    }
    
	// Mettre à jour p[]
    for (i=0; i<n;i++){
	    	p[i] = p_new[i];
    }
    
	// Augmenter le nombre d'itérations
    k=k+2;
}
  
  /*************************** CONCLUSION *******************************/

  // Arrête le chronomètre et calcule le temps passé
  end = clock();
  time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
  
  // Sleep un peu pour que la sortie standard ne soit pas foirée
  //Sleep(500);
    
  // Imprimer les résultats
  printf ("\nNombre diterations pour converger: %d \n\n", k); 
  printf ("Valeurs finales du PageRank:\n\n[");
  for (i=0; i<n; i++){
    printf("%f ", p[i]);
    if(i!=(n-1)){ printf(", "); }
  }
  
	printf("\n\n\n");
	printf("                ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	Sleep(100);
	printf("                                     Temps passe: %f secondes.	                    \n",time_spent);
	Sleep(100);
	printf("                                     Iterations: %d \n	                      				    \n",k);
	Sleep(100);
	printf("                +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n");
	system("color 2");
	printf("                                               ./oydmNMMMMNmdyo/`\n");
	Sleep(100);
	printf("                                            -smMMMMMMMMMMMMMMMMMMms:\n");
	Sleep(100);
	printf("                                          +mMMMMMMMMMMMMMMMMMMMMMMMMm\n");
	Sleep(100);
	printf("                                        /NMMMMMMMMMMMMMMMMMMMMMMMMMMMMN/\n");
	Sleep(100);
	printf("                                      `hMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMh`\n");
	Sleep(100);
	printf("                                     .mMMMMMMMMMMMMMMMMMMMMMMMMmdmMMMMMMMm.\n");
	Sleep(100);
	printf("                                     dMMMMMMMMMMMMMMMMMMMMMMMh-   -NMMMMMMd\n");
	Sleep(100);
	printf("                                    +MMMMMMMMMMMMMMMMMMMMMMh-      dMMMMMMM+\n");
	Sleep(100);
	printf("                                    mMMMMMMMMNhshNMMMMMMMh-      :dMMMMMMMMN\n");
	Sleep(100);
	printf("                                    MMMMMMMMm`   `oNMMMh-      :mMMMMMMMMMMM\n");
	Sleep(100);
	printf("                                    MMMMMMMMm.     `oh-      -hMMMMMMMMMMMMM\n");
	Sleep(100);
	printf("                                    NMMMMMMMMNs`           -hMMMMMMMMMMMMMMm\n");
	Sleep(100);
	printf("                                    +MMMMMMMMMMNs`       :hMMMMMMMMMMMMMMMM+\n");
	Sleep(100);
	printf("                                     dMMMMMMMMMMMNs`   /mMMMMMMMMMMMMMMMMMd\n");
	Sleep(100);
	printf("                                     .mMMMMMMMMMMMMMddNMMMMMMMMMMMMMMMMMMm.\n");
	Sleep(100);
	printf("                                      `hMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMh`\n");
	Sleep(100);
	printf("                                        /NMMMMMMMMMMMMMMMMMMMMMMMMMMMMN/\n");
	Sleep(100);
	printf("                                          +mMMMMMMMMMMMMMMMMMMMMMMMMm+\n");
	Sleep(100);
	printf("                                            -smMMMMMMMMMMMMMMMMMMms-\n");
	Sleep(100);
	printf("                                               `:oydmNMMMMNmdyo:`\n\n");
  return 0;
}
