namespace TestProject
{
    public class Misformatted
{
            public int Value;
        public Misformatted(int value){
    Value=value;
        }

    public int Double( )
        {
    return Value*2;
    }


        public   string   Describe()
    {
        if(Value>0){
    return "positive";
            }else{
        return "non-positive";
    }
    }
    }
}
