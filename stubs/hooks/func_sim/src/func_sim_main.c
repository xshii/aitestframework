extern void platform_sim_setup(void);
extern int  stub_entry(int argc, char *argv[]);

int main(int argc, char *argv[])
{
    platform_sim_setup();
    return stub_entry(argc, argv);
}
