**Sub-directory for sharing student-created data**

**Note:** Please save `.Rds` files whenever possible, and edit this README.md to explain what your file(s) contain!

To **save** a dataframe `df` as a Rds file in your current working directory:

   * _generate your dataframe (let's call it "df")_ 
   * `saveRDS(df, "df.Rds")`


To **create a dataframe** by **reading** an Rds file in your current working directory: 

   * `df <- readRDS("df.Rds")`

_You may need to use a more creative path name to read an Rds in the StudentData sub-directory..._
